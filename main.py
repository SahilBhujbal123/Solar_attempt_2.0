import numpy as np
from PIL import Image
import io
import keras
from pathlib import Path
from fastapi import FastAPI, UploadFile, File


app = FastAPI()

imageshape = (130, 130)

BASE_DIR = Path(__file__).resolve(strict=True).parent 
cnn = keras.models.load_model(BASE_DIR / "130pxCNN.keras")

def read_image(image_encoded: bytes, imageshape: tuple) -> np.array:

    # Open the image from bytes
    image = Image.open(io.BytesIO(image_encoded))
    # Convert image to numpy array
    image = np.array(image)
    if image is None:
        raise ValueError("Failed to read image.")
    # preprocess
    image = image[:, :, :3]                            
    image = np.array(Image.fromarray(image).convert('RGB'))  
    image = np.array(Image.fromarray(image).resize(imageshape))
    
    return image / 255
 

def predict_solar(image: np.array) -> np.array:
    prediction = cnn.predict(np.expand_dims(image, 0))
    return prediction

@app.get("/")
async def home():
    return {"message": "Welcome to the API!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png", "PNG", "JPG", "JPEG")
    if not extension:
        return "Image must be jpg or png format!"
    try:
        image = read_image(await file.read(), imageshape)
        prediction = predict_solar(image)
        return {"prediction": prediction[0][0].tolist()}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8080, host='0.0.0.0')
