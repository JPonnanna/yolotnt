from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
from fastapi.responses import JSONResponse
import cv2
import numpy as np

app = FastAPI()
model = YOLO("best.pt")  # Adjust if needed based on folder structure

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    results = model(img)[0]

    predictions = []
    for box in results.boxes:
        predictions.append({
            "class_id": int(box.cls[0]),
            "confidence": float(box.conf[0]),
            "bbox": [int(x) for x in box.xyxy[0]]
        })

    return JSONResponse(content=predictions)
