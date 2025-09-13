# python_service/main.py
# --- MODIFIED IMPORTS ---
from fastapi import FastAPI, UploadFile, File, BackgroundTasks # Add BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import base64
from ultralytics import YOLO
import uvicorn
from PIL import Image
import io
from datetime import datetime # Add datetime for timestamping
from smtp_server import send_alert_email # Import our new function
import time

app = FastAPI()

# --- Add CORS (No changes here) ---
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your YOLO model once (No changes here)
MODEL_PATH = "best_poacher.pt"
model = YOLO(MODEL_PATH)

def encode_image_to_base64(img_bgr): # (No changes here)
    ret, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ret:
        raise RuntimeError("Failed to encode image")
    return base64.b64encode(buf.tobytes()).decode("utf-8")

# --- MODIFIED PREDICT ENDPOINT ---
@app.post("/predict")
async def predict(file: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    try:
        # Read image from uploaded file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        frame_np = np.array(image)

        # Convert RGB → BGR for OpenCV/YOLO
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)

        # Run YOLO prediction
        results = model(frame_bgr, conf=0.25)

        # Get annotated image (RGB from ultralytics)
        annotated = results[0].plot()

        # Convert back to BGR for encoding
        annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

        # Encode to base64 for JSON response
        annotated_b64 = encode_image_to_base64(annotated_bgr)

        # Collect detections
        detections = []
        boxes = results[0].boxes
        if boxes is not None:
            # --- NEW: Prepare image bytes for potential email attachment ---
            # We do this once before the loop for efficiency
            ret, buf = cv2.imencode(".jpg", annotated_bgr)
            image_bytes_for_email = buf.tobytes() if ret else None

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

                # --- NEW: TRIGGER EMAIL ALERT ON DETECTION ---
                if label.lower() in ["weapon", "poacher"] and image_bytes_for_email:
                    print(f"Threat detected: '{label}'. Adding email alert to background tasks.")
                    # Add the email sending task to run in the background
                    background_tasks.add_task(
                        send_alert_email,
                        image_bytes=image_bytes_for_email,
                        label=label,
                        confidence=conf,
                        timestamp=datetime.now()
                    )
                    time.sleep(15)

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