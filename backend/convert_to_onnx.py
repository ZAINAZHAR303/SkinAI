"""
Run this ONCE locally to convert the PyTorch model to ONNX format.
Then upload convolutional_network.onnx to Hugging Face:

    huggingface-cli upload zainazhar303/skin-disease-cnn convolutional_network.onnx
"""

import torch
import torch.nn as nn


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
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


model = SkinDiseaseCNN()
model.load_state_dict(torch.load("convolutional_network.pth", map_location="cpu"))
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    "convolutional_network.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=11,
    dynamo=False,  # use legacy exporter which embeds weights
)

import os
size_mb = os.path.getsize("convolutional_network.onnx") / (1024 * 1024)
print(f"✅ Saved convolutional_network.onnx ({size_mb:.1f} MB)")
if size_mb < 10:
    print("⚠️  File looks too small — weights may not be embedded. Check export.")
else:
    print("Now run: huggingface-cli upload zainazhar303/skin-disease-cnn convolutional_network.onnx")
