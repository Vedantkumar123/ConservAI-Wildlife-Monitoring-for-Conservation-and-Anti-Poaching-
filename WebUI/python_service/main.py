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
# ✅ Import the new tracker
from tracker import ObjectTracker

# -------------------------
# FastAPI App Initialization
# -------------------------
app = FastAPI()

# --- Add CORS (No changes) ---
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
# Load YOLO Model & Tracker
# -------------------------
MODEL_PATH = "best_poacher.pt"
model = YOLO(MODEL_PATH)
# ✅ Create a single, global instance of the tracker
tracker = ObjectTracker()

# ✅ Global variable to store latest state (detections + counts)
latest_state = {
    "detections": [],
    "counts": {}
}

# --- Helper Function (No changes) ---
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
    global latest_state
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        frame_np = np.array(image)
        # Convert to BGR for OpenCV drawing
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)

        # Run YOLO prediction
        results = model(frame_np, conf=0.25)

        # ✅ Collect raw detections for the tracker
        raw_detections = []
        boxes = results[0].boxes
        if boxes is not None:
            for box in boxes:
                raw_detections.append({
                    "box": box.xyxy[0].tolist(),
                    "confidence": float(box.conf[0]),
                    "class_id": int(box.cls[0]),
                    "label": model.names[int(box.cls[0])]
                })

        # ✅ Get tracked objects with unique IDs
        tracked_detections = tracker.update(raw_detections)
        active_counts = tracker.get_active_counts()
        
        # ✅ Annotate the image with TRACKED detections
        annotated_frame_bgr = frame_bgr.copy()
        image_bytes_for_email = None

        if tracked_detections:
            for det in tracked_detections:
                box = det['box']
                # Create label with unique ID, e.g., "tiger-1 (0.95)"
                unique_label = f"{det['unique_id']} ({det['confidence']:.2f})"
                x1, y1, x2, y2 = [int(coord) for coord in box]
                
                # Draw bounding box and label
                cv2.rectangle(annotated_frame_bgr, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(annotated_frame_bgr, unique_label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # --- Encode annotated image ---
            annotated_b64 = encode_image_to_base64(annotated_frame_bgr)
            
            # --- Get image bytes for email ---
            ret, buf = cv2.imencode(".jpg", annotated_frame_bgr)
            image_bytes_for_email = buf.tobytes() if ret else None

            # --- Handle Email Alerts ---
            if image_bytes_for_email:
                for det in tracked_detections:
                    if det['label'].lower() in ["weapon", "poacher"]:
                        print(f"🚨 Threat detected: '{det['label']}'. Email alert scheduled.")
                        background_tasks.add_task(
                            send_alert_email,
                            image_bytes=image_bytes_for_email,
                            label=det['label'],
                            confidence=det['confidence'],
                            timestamp=datetime.now()
                        )
        else:
            # No detections, just encode the original frame
            annotated_b64 = encode_image_to_base64(frame_bgr)

        # ✅ Update the global state
        latest_state = {
            "detections": tracked_detections,
            "counts": active_counts
        }

        return JSONResponse({
            "status": "ok",
            "detections": tracked_detections,
            "annotated_image_b64": annotated_b64,
            "active_counts": active_counts # ✅ Send counts back
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ Error in /predict: {e}")
        return JSONResponse(
            {"status": "error", "message": str(e)}, status_code=500
        )


# -------------------------
# Get Latest Detection Endpoint
# -------------------------
@app.get("/predict-latest")
async def predict_latest():
    # ✅ Return the full latest state including counts
    return JSONResponse({
        "status": "ok",
        "detections": latest_state["detections"],
        "counts": latest_state["counts"]
    })


# -------------------------
# Run Server
# -------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)