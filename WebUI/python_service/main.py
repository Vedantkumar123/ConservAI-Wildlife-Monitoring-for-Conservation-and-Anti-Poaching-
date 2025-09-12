# python_service/main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import base64
from ultralytics import YOLO
import uvicorn
from PIL import Image
import io

app = FastAPI()

# --- Add CORS ---
origins = [
    "http://localhost:3000",   # React dev server
    "http://127.0.0.1:3000",   # Alternate localhost
    # "http://yourdomain.com"  # add if you deploy frontend separately
]

app.add_middleware(
    CORSMiddleware,
    # allow_origins=origins,       # or ["*"] to allow all origins (not safe for prod)
    allow_origins=["*"],   # or ["http://localhost:3000"] for stricter security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your YOLO model once
MODEL_PATH = "best_poacher.pt"
model = YOLO(MODEL_PATH)

def encode_image_to_base64(img_bgr):
    ret, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ret:
        raise RuntimeError("Failed to encode image")
    return base64.b64encode(buf.tobytes()).decode("utf-8")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image from uploaded file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        frame_np = np.array(image)  # RGB numpy array

        # Convert RGB → BGR for OpenCV/YOLO
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)

        # Run YOLO prediction
        results = model(frame_bgr, conf=0.25)

        # Get annotated image (RGB from ultralytics)
        annotated = results[0].plot()

        # Convert back to BGR for encoding
        annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

        # Encode to base64
        annotated_b64 = encode_image_to_base64(annotated_bgr)

        # Collect detections
        detections = []
        boxes = results[0].boxes
        if boxes is not None:
            for box in boxes:
                xyxy = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = model.names[cls]
                detections.append({
                    "box": xyxy,
                    "confidence": conf,
                    "class_id": cls,
                    "label": label
                })

        return JSONResponse({
            "status": "ok",
            "detections": detections,
            "annotated_image_b64": annotated_b64
        })

    except Exception as e:
        return JSONResponse(
            {"status": "error", "message": str(e)}, status_code=500
        )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
