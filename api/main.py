from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure the model path is correct
model_path = os.path.abspath("../saved_models/1")
logger.info(f"Loading model from {model_path}")

try:
    MODEL = tf.keras.models.load_model(model_path)
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise e

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]


@app.get("/ping")
async def ping():
    return {"message": "Hello, I am alive"}


def read_file_as_image(data) -> np.ndarray:
    try:
        image = np.array(Image.open(BytesIO(data)))
        return image
    except Exception as e:
        logger.error(f"Error reading image: {e}")
        raise e


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        logger.info("Reading image...")
        image = read_file_as_image(await file.read())
        img_batch = np.expand_dims(image, 0)

        logger.info("Making prediction...")
        predictions = MODEL.predict(img_batch)

        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])
        return {
            'class': predicted_class,
            'confidence': float(confidence)
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise e


if __name__ == "__main__":
    logger.info("Starting server...")
    try:
        uvicorn.run(app, host='localhost', port=8000)
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise e
