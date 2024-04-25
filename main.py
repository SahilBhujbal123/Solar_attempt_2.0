import numpy as np
import io
import keras
from pathlib import Path
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging

# Initialize the app and logger
app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust to the front-end URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants and model loading
imageshape = (200, 200)
BASE_DIR = Path(__file__).resolve(strict=True).parent
cnn = keras.models.load_model(BASE_DIR / "200pxmodel.keras")

def read_image(image_encoded: bytes, imageshape: tuple) -> np.array:
    try:
        with Image.open(io.BytesIO(image_encoded)) as image:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image = image.resize(imageshape, Image.BILINEAR)
            return np.array(image) / 255
    except Exception as e:
        logger.error(f"Invalid image file: {str(e)}")
        raise ValueError(f"Invalid image file: {str(e)}")

def predict_solar(image: np.array) -> np.array:
    prediction = cnn.predict(np.expand_dims(image, 0))
    return prediction

@app.get("/")
async def home():
    return {"message": "Welcome to the API! 150px_3"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    valid_extensions = ("jpg", "jpeg", "png")
    extension = file.filename.split(".")[-1].lower() in valid_extensions
    if not extension:
        return {"error": "Image must be jpg or png format!"}

    if file.size > 10_000_000:  # Limit file size to 10 MB
        raise HTTPException(status_code=413, detail="File too large")

    try:
        image_data = await file.read()
        image = read_image(image_data, imageshape)
        prediction = predict_solar(image)
        return {"prediction": prediction[0][0].tolist()}
    except ValueError as e:
        logger.error(f"Error processing image: {str(e)}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return {"error": f"An error occurred: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8080, host='0.0.0.0')
