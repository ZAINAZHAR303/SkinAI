from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

from model import predict_image

app = FastAPI(title="Skin Disease Detection API")

# ✅ allow Next.js
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later restrict
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    result = predict_image(image)
    return result