import os
import urllib.request
import numpy as np
import onnxruntime as ort
from PIL import Image

# 📥 Auto-download ONNX model from Hugging Face if not present locally
# Using ONNX + onnxruntime instead of PyTorch to stay within Render's 512MB RAM limit
MODEL_PATH = "convolutional_network.onnx"
MODEL_URL  = "https://huggingface.co/zainazhar303/skin-disease-cnn/resolve/main/convolutional_network.onnx"

if not os.path.exists(MODEL_PATH):
    print("Model not found locally. Downloading from Hugging Face...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("✅ Model downloaded successfully.")

# Load ONNX session once at startup (~30MB RAM vs ~300MB for PyTorch)
print("Loading ONNX model...")
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
INPUT_NAME  = session.get_inputs()[0].name
OUTPUT_NAME = session.get_outputs()[0].name
print("✅ Model ready.")

# Preprocessing — matches training transforms
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def _preprocess(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize((224, 224))
    arr = np.array(image, dtype=np.float32) / 255.0   # → [0, 1]
    arr = (arr - _MEAN) / _STD                         # normalize
    arr = arr.transpose(2, 0, 1)                       # HWC → CHW
    return arr[np.newaxis, :]                          # add batch dim

def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()

CLASS_NAMES = [
    "Melanoma",
    "Melanocytic Nevus",
    "Basal Cell Carcinoma",
    "Actinic Keratosis",
    "Benign Keratosis",
    "Dermatofibroma",
    "Vascular Lesion",
    "Squamous Cell Carcinoma",
    "Unknown"
]

def predict_image(image: Image.Image):
    input_tensor = _preprocess(image)
    outputs = session.run([OUTPUT_NAME], {INPUT_NAME: input_tensor})[0][0]
    probs = _softmax(outputs)
    pred = int(np.argmax(probs))
    confidence = float(probs[pred])
    return {
        "class": CLASS_NAMES[pred],
        "confidence": round(confidence * 100, 2)
    }