## 🧠 Model Architecture: Custom-Optimized YOLOv11

The core of **ConservAI** is a **custom-modified YOLOv11 detector** designed specifically for wildlife monitoring and anti-poaching scenarios, where standard object detectors struggle due to **occlusion, scale variation, cluttered backgrounds, and low-visibility conditions**.

While YOLOv11 provides a strong real-time baseline, its default configuration is limited in **small-object recall**, **contextual reasoning across scales**, and **robustness to partial occlusion**. To address these limitations, the architecture was extended across the **backbone, neck, and feature fusion stages**, while preserving real-time inference capability for edge deployment.

---

### 🔹 High-Level Architecture Overview

The custom architecture follows a **Backbone → Neck → Detection Head** design, with targeted enhancements at each stage:

```
Input Image
   ↓
Enhanced Backbone
(CNN + Attention + Global Context)
   ↓
BiFPN + Deformable Fusion Neck
   ↓
YOLOv11 Detection Head
   ↓
Detections → Analytics → Alerts
```

Each architectural modification is motivated by a **specific real-world failure mode** commonly observed in forest surveillance footage.

---

## 🧩 Backbone: Rich Feature Extraction with Attention & Global Context

The backbone is responsible for extracting hierarchical visual features from input frames. In wildlife environments, this stage must capture **subtle texture cues**, **partial silhouettes**, and **low-contrast objects** against complex natural backgrounds.

### 🔸 Baseline Feature Extraction

Early layers retain YOLOv11’s convolutional and **C3k2 partial bottleneck blocks**, which efficiently extract low-level spatial features such as edges, contours, and textures.

### 🔸 Channel & Spatial Attention (CBAM)

To suppress background noise and emphasize foreground objects, **Convolutional Block Attention Modules (CBAM)** are introduced at multiple backbone stages.

CBAM applies:

* **Channel Attention** to identify *what* features are important
* **Spatial Attention** to identify *where* they occur

This is especially effective in:

* Dense vegetation
* Camouflaged animals
* Low-contrast lighting conditions

### 🔸 Global Context via MobileViT Blocks

Standard YOLO architectures rely heavily on local receptive fields. To improve **long-range dependency modeling**, **MobileViT blocks** are incorporated in intermediate backbone layers.

These transformer-inspired blocks:

* Capture global scene context
* Improve detection of partially visible or distant targets
* Help distinguish humans from background clutter over large spatial extents

### 🔸 Multi-Scale Context Enhancement

To further improve scale robustness, the backbone integrates:

* **SPPF (Spatial Pyramid Pooling – Fast)** for efficient receptive field expansion
* **ASPP (Atrous Spatial Pyramid Pooling)** for capturing features at multiple dilation rates

This combination enables the network to detect:

* Small prey animals
* Large mammals
* Human intruders at varying distances

### 🔸 High-Level Semantic Refinement

At the top of the backbone:

* **CBAM** and **C2PSA (Cross-Stage Partial Self-Attention)** modules refine high-level features
* Foreground regions are emphasized while irrelevant background activations are suppressed

This improves robustness under:

* Motion blur
* Partial occlusion
* Low-light conditions

---

## 🔗 Neck: BiFPN with Deformable Convolutions & Attention

The neck is responsible for **multi-scale feature fusion**, which is critical when detecting objects of drastically different sizes within the same frame.

### 🔸 Bi-Directional Feature Pyramid Network (BiFPN)

The standard PAN/FPN neck is replaced with a **BiFPN**, which introduces:

* Learnable weighted feature fusion
* Bidirectional (top-down + bottom-up) information flow
* Improved cross-scale consistency

BiFPN allows the model to dynamically balance semantic and spatial information across feature levels, significantly improving small-object detection without excessive computational overhead.

### 🔸 Deformable Convolutions (DCN)

After each BiFPN fusion stage, **Deformable Convolutions** are applied.

DCN enables:

* Adaptive receptive fields
* Better alignment with irregular object shapes
* Improved localization of partially visible animals and humans

This is particularly effective for:

* Animals in non-rigid postures
* Poachers partially hidden behind foliage or terrain

### 🔸 Attention-Refined Feature Fusion

Following deformable fusion, **CBAM attention modules** are applied to:

* Strengthen foreground activations
* Suppress background interference
* Improve localization accuracy

The combined BiFPN + DCN + Attention design ensures **robust contextual reasoning** while preserving inference speed.

---

## 🎯 Detection Head & Optimization

The detection head remains structurally similar to YOLOv11 but benefits directly from the enhanced feature representations produced by the backbone and neck.

### 🔹 Resulting Improvements

* Higher confidence in small and occluded detections
* Reduced false positives in cluttered scenes
* Improved bounding box localization across scales

### 🔹 Performance Optimization

Although the added modules increase architectural complexity, real-time performance is recovered using:

* **TensorRT acceleration**
* **FP16 / INT8 quantization**
* **Convolution + BatchNorm fusion**

These optimizations restore near-baseline inference speed while retaining accuracy gains.

---

## ⚡ Edge Deployment Readiness

The final architecture is explicitly designed for **edge deployment in remote wildlife environments**, balancing:

* Accuracy
* Latency
* Memory footprint

The optimized model achieves:

* ~110 FPS inference
* Low latency suitable for real-time alerting
* Stable performance under field conditions

---

## 🧠 Architectural Impact

Overall, the architectural enhancements deliver:

* **5–6% mAP improvement** over baseline YOLOv11
* Significant gains in **small-object recall**
* Robust performance under occlusion and complex backgrounds

This makes the model well-suited for **continuous wildlife monitoring, anti-poaching surveillance, and real-time conservation intelligence systems**.

---
## 📊 Analytics Framework

Detection outputs are streamed into Python pipelines and visualized using **Power BI**, enabling both **ecological monitoring** and **anti-poaching intelligence**.

### 🌿 Wildlife Activity Metrics

* **Species Activity Index (SAI)**
* **Diel Activity Ratio (DAR)**
* **Habitat Usage Ratio (HUR)**
* **Species Co-occurrence Matrix**
* **Rolling Average Activity Trends**

### 🚨 Security & Anti-Poaching Metrics

* **Poaching Risk Index (PRI)**
* **Poacher–Ranger Interaction Rate (PRIR)**
* **Average Response Time (ART)**
* **Patrol Efficiency Index (PEI)**
* **Kernel Density Estimation (KDE) hotspot maps**

These metrics allow authorities to identify **high-risk zones**, **optimize patrol routes**, and **measure operational effectiveness**.

---

## ⚙️ Automated Alert System

* Real-time human/poacher detection triggers alerts
* SMTP-based notifications to conservation authorities
* Designed for rapid intervention and reduced response latency

---

## 📈 Results Summary

| Metric        | Value |
| ------------- | ----- |
| mAP@0.5       | ~84%  |
| mAP@0.5:0.95  | ~64%  |
| Precision     | 0.87  |
| Recall        | 0.83  |
| Optimized FPS | ~110  |

The most significant gains were observed in **small-object detection** and **occlusion-heavy scenarios**, common in forest environments.

---

## ⚠️ Limitations

* Reliance on RGB cameras limits night-time performance
* Network instability in remote forest regions
* Species diversity challenges large-scale classification
* Potential overcounting without multi-object tracking

These limitations are actively considered in future work.

---

## 🔮 Future Work

* Integration of **thermal / infrared sensors** for night surveillance
* Advanced **threat differentiation** (weapons, vehicles, behavior cues)
* Store-and-forward and mesh networking for low-connectivity regions
* Active learning using ranger feedback for continuous model improvement
* Object tracking to mitigate overcounting and improve temporal reasoning



Just say the word.
