import torch
import torch.nn as nn

class Channel_Attention(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(channels,channels,1,1,0,bias=True)
        self.act = nn.Sigmoid()

    def forward(self,x):

        return x*self.act(self.conv(self.pool(x)))
    
class Spatial_Attention(nn.Module):

    def __init__(self, kernal_size = 7):
        super().__init__()
        assert kernal_size in {3,7}, "kernal size must be (3 or 7)"
        padding = 3 if kernal_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size=kernal_size,padding=padding, bias=False)
        self.act = nn.Sigmoid()
    
    def forward(self, x):
        avg_x = torch.mean(x, 1, keepdim=True)
        max_x = torch.max(x, 1, keepdim=True)[0]
        cat_x = torch.cat([avg_x,max_x],1)

        return x * self.act(self.conv(cat_x))

class CBAM(nn.Module):

    def __init__(self, c1, kernal_size=7):
        super().__init__()
        self.channel_attention = Channel_Attention(c1)
        self.spatial_attention = Spatial_Attention(kernal_size=kernal_size)

    def forward(self, x):

        return self.spatial_attention(self.channel_attention(x))


class ASPP_Lite(nn.Module):
    
    def __init__(self,c1,dilations=(1, 6, 12, 18), reduction=4):
        super().__init__()
        c2 = c1
        c_ = c2 // reduction
        self.branch1 = nn.Sequential(
            nn.Conv2d(c1, c_, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_),
            nn.ReLU(inplace=True)
        )
        self.branches = nn.ModuleList()
        for d in dilations:
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(c1, c_, kernel_size=3, padding=d, dilation=d, bias=False),
                    nn.BatchNorm2d(c_),
                    nn.ReLU(inplace=True)
                )
            )
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_conv = nn.Sequential(
            nn.Conv2d(c1, c_, kernel_size=1, bias=False),
            # nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True)
        )
        
        self.project = nn.Sequential(
            nn.Conv2d((len(dilations)+2)*c_, c2 ,kernel_size=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True)
        )
    
    def forward(self,x):
        h, w = x.shape[2],x.shape[3]
        out1 = self.branch1(x)
        outs = [out1]

        for branch in self.branches:
            outs.append(branch(x))
        
        gp = self.global_pool(x)
        gp = self.global_conv(gp)
        gp = nn.functional.interpolate(gp, size=(h,w), mode='bilinear', align_corners=False)
        outs.append(gp)

        out = torch.cat(outs, dim=1)
        return self.project(out)
    

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ConvBNAct(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1, bias=False, act=True):
        layers = [nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=bias),
                  nn.BatchNorm2d(out_ch)]
        if act:
            layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)


class MobileViTBlock(nn.Module):

    def __init__(
        self,
        c1: int,
        depth: int = 2,
        patch_size: int = 2,
        expansion: int = 2,
        num_heads: Optional[int] = None,
        mlp_ratio: int = 2,
    ):
        super().__init__()
        assert patch_size >= 1 and isinstance(patch_size, int)
        self.c_in = c1
        self.depth = depth
        self.patch_size = patch_size
        # c_local = int(min( c1 * expansion, 2048 ))  # small expansion
        c_local = c1//4
        self.local_feat = ConvBNAct(c1, c_local, kernel=3, padding=1)
        self.project_to_tokens = nn.Sequential(
            nn.Conv2d(c_local, c_local, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_local),
            nn.ReLU(inplace=True),
        )
        embed_dim = c_local
        
        if num_heads is None:
           
            if embed_dim % 8 == 0:
                num_heads = 8
            elif embed_dim % 4 == 0:
                num_heads = 4
            else:
               
                num_heads = 1
        self.num_heads = num_heads
       
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * mlp_ratio,
            batch_first=True,
            activation='relu',
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        self.project_from_tokens = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
       
        self.fusion = nn.Sequential(
            nn.Conv2d(c_local + embed_dim, c1, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            ConvBNAct(c1, c1, kernel=3, padding=1)  
        )

    def _pad_to_multiple(self, x, multiple):
        
        _, _, h, w = x.shape
        pad_h = (multiple - (h % multiple)) % multiple
        pad_w = (multiple - (w % multiple)) % multiple
        if pad_h == 0 and pad_w == 0:
            return x, (0, 0)
        
        x = F.pad(x, (0, pad_w, 0, pad_h))
        return x, (pad_h, pad_w)

    def forward(self, x):
        B, C, H, W = x.shape        
        local = self.local_feat(x) 
        proj = self.project_to_tokens(local)  
        proj, (pad_h, pad_w) = self._pad_to_multiple(proj, self.patch_size)
        _, Cproj, Hp, Wp = proj.shape
        Ph = self.patch_size
        Pw = self.patch_size
        nh = Hp // Ph
        nw = Wp // Pw
        num_patches = nh * nw
        proj = proj.view(B, Cproj, nh, Ph, nw, Pw)
        proj = proj.permute(0, 2, 4, 3, 5, 1).contiguous()  
        proj = proj.view(B, num_patches, Ph * Pw * Cproj) 
        if proj.shape[-1] != Cproj:
            
            proj = proj.view(B * num_patches, -1)
            proj = nn.functional.adaptive_avg_pool1d(proj.unsqueeze(1), Cproj).squeeze(1)
            proj = proj.view(B, num_patches, Cproj)
        
        tokens = proj  
        tokens = self.transformer(tokens)  
        tokens = tokens.view(B, nh, nw, Cproj) 
        tokens = tokens.permute(0, 3, 1, 2).contiguous()  # (B, embed_dim, nh, nw)
        tokens = tokens.unsqueeze(-1).unsqueeze(-1)  
        tokens = tokens.repeat(1, 1, 1, 1, Ph, Pw)  
        tokens = tokens.permute(0, 1, 2, 4, 3, 5).contiguous() 
        tokens = tokens.view(B, Cproj, Hp, Wp) 
        tokens = self.project_from_tokens(tokens)  

        if pad_h != 0 or pad_w != 0:
            tokens = tokens[:, :, :H, :W]
            local = local[:, :, :H, :W]
        fused = torch.cat([local, tokens], dim=1)
        out = self.fusion(fused)
        if out.shape == x.shape:
            out = out + x

        return out

if __name__ == "__main__":

    ### CBAM
    cbam = CBAM(c1=64, kernal_size=7)
    print(f"CBAM {sum(p.numel() for p in cbam.parameters())/1e6} million parameters")    
    dummy_input = torch.rand((1,64,256,256))
    dummy_output = cbam(dummy_input)
    print("CBAM input shape: ",dummy_input.shape)
    print("CBAM Output shape: ",dummy_output.shape)

    ### ASPP
    aspp = ASPP_Lite(c1=1024)
    print(f"ASPP Parameters: {sum(p.numel() for p in aspp.parameters())/1e6:.2f}M")
    dummy = torch.randn(1, 1024, 32, 32)
    out = aspp(dummy)
    print("ASPP_Lite Input shape :", dummy.shape)
    print("ASPP_Lite Output shape:", out.shape)

    ### MobileViTBlock
    mvit = MobileViTBlock(512, depth=2, patch_size=2)
    print(f"MobileViTBlock Parameters: {sum(p.numel() for p in mvit.parameters())/1e6:.3f}M")
    dummy = torch.randn(1, 512, 32, 32)
    out = mvit(dummy)
    print("MobileViTBlock Input shape :", dummy.shape)
    print("MobileViTBlock Output shape:", out.shape)