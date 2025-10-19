# python_service/main.py

# --- MODIFIED IMPORTS ---
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import base64
from ultralytics import YOLO
import uvicorn
from PIL import Image
import io
from datetime import datetime

# --- LOCAL IMPORTS ---
from smtp_server import send_alert_email
# Import the new function to handle saving detections
from database_utils import save_detection

# -------------------------
# FastAPI App Initialization
# -------------------------
app = FastAPI()

# --- Add CORS ---
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

# -------------------------
# Load YOLO Model
# -------------------------
MODEL_PATH = "best_poacher.pt"
model = YOLO(MODEL_PATH)

# --- Global variable to store latest detections for the polling endpoint ---
latest_detections = []

# --- Helper Function ---
def encode_image_to_base64(img_bgr):
    """Convert OpenCV image to base64 string"""
    ret, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ret:
        raise RuntimeError("Failed to encode image")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


# -------------------------
# Prediction Endpoint
# -------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    global latest_detections
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        frame_np = np.array(image)

        # Run YOLO prediction
        results = model(frame_np, conf=0.25)
        annotated = results[0].plot()
        annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        annotated_b64 = encode_image_to_base64(annotated_bgr)

        current_frame_detections = []
        boxes = results[0].boxes
        if boxes is not None:
            ret, buf = cv2.imencode(".jpg", annotated_bgr)
            image_bytes_for_email = buf.tobytes() if ret else None

            for box in boxes:
                xyxy = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = model.names[cls]

                # --- Prepare Detection Object ---
                # This dictionary holds all the information for the database
                detection_data = {
                    "Animal": label,
                    "Cam_id": "Cam1",          # Replace with dynamic ID if available
                    "Location": "Region A",      # Replace with real location if available
                    "Severity": "Vulnerable" if label.lower() == "tiger" else "Threat" if label.lower() in ["weapon", "poacher"] else "Monitored",
                    # The timestamp will be set inside the save function for accuracy
                }

                # --- SAVE TO DATABASE (with duplicate check) ---
                # Call the refactored function from database_utils.py
                save_detection(detection_data)

                # Append details for the API response
                current_frame_detections.append({
                    "box": xyxy,
                    "confidence": conf,
                    "class_id": cls,
                    "label": label
                })

                # --- TRIGGER EMAIL ALERT ---
                if label.lower() in ["weapon", "poacher"] and image_bytes_for_email:
                    print(f"🚨 Threat detected: '{label}'. Email alert scheduled.")
                    background_tasks.add_task(
                        send_alert_email,
                        image_bytes=image_bytes_for_email,
                        label=label,
                        confidence=conf,
                        timestamp=datetime.now()
                    )

        latest_detections = current_frame_detections

        return JSONResponse({
            "status": "ok",
            "detections": current_frame_detections,
            "annotated_image_b64": annotated_b64
        })

    except Exception as e:
        print(f"❌ Error in /predict: {e}")
        return JSONResponse(
            {"status": "error", "message": str(e)}, status_code=500
        )


# -------------------------
# Get Latest Detection Endpoint
# -------------------------
@app.get("/predict-latest")
async def predict_latest():
    return JSONResponse({
        "status": "ok",
        "detections": latest_detections
    })


# -------------------------
# Run Server
# -------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)