import os
import urllib.request
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image


MODEL_PATH = "convolutional_network.pth"
MODEL_URL  = "https://huggingface.co/zainazhar303/skin-disease-cnn/resolve/main/convolutional_network.pth"

if not os.path.exists(MODEL_PATH):
    print(f"Model not found locally. Downloading from Hugging Face...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print(" Model downloaded successfully.")

# Architecture reconstructed from convolutional_network.pth weight shapes:
# conv1: Conv2d(3, 16, 3) | conv2: Conv2d(16, 32, 3)
# fc1: Linear(100352, 256) | fc2: Linear(256, 64) | fc3: Linear(64, 9)
class SkinDiseaseCNN(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.relu  = nn.ReLU()
        self.fc1   = nn.Linear(32 * 56 * 56, 256)
        self.fc2   = nn.Linear(256, 64)
        self.fc3   = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # → 16 x 112 x 112
        x = self.pool(self.relu(self.conv2(x)))  # → 32 x 56  x 56
        x = x.view(x.size(0), -1)               # → 100352
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SkinDiseaseCNN(num_classes=9)
model.load_state_dict(torch.load("convolutional_network.pth", map_location=device))
model.to(device)
model.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


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
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

    return {
        "class": CLASS_NAMES[pred],
        "confidence": round(confidence * 100, 2)
    }