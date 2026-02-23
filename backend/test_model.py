import torch
import torch.nn as nn

CLASS_NAMES = [
    "Melanoma", "Melanocytic Nevus", "Basal Cell Carcinoma",
    "Actinic Keratosis", "Benign Keratosis", "Dermatofibroma",
    "Vascular Lesion", "Squamous Cell Carcinoma", "Unknown"
]

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

torch.manual_seed(42)
img = torch.randn(1, 3, 224, 224)

# --- Untrained (pure random) ---
rand_m = SkinDiseaseCNN(); rand_m.eval()
with torch.no_grad():
    pr = torch.softmax(rand_m(img), dim=1)[0]
print("=" * 50)
print("UNTRAINED (random weights)")
print(f"  Max confidence : {pr.max()*100:.2f}%")
print(f"  Std of probs   : {pr.std()*100:.4f}%")
for n, p in zip(CLASS_NAMES, pr):
    print(f"  {n:<30s} {p*100:.2f}%")

# --- Your trained model ---
trained = SkinDiseaseCNN()
trained.load_state_dict(torch.load("convolutional_network.pth", map_location="cpu"))
trained.eval()
with torch.no_grad():
    pt = torch.softmax(trained(img), dim=1)[0]
print()
print("=" * 50)
print("YOUR TRAINED MODEL")
print(f"  Max confidence : {pt.max()*100:.2f}%")
print(f"  Std of probs   : {pt.std()*100:.4f}%")
for n, p in zip(CLASS_NAMES, pt):
    print(f"  {n:<30s} {p*100:.2f}%")

# --- Consistency: same input -> same output every time ---
with torch.no_grad():
    pt2 = torch.softmax(trained(img), dim=1)[0]
print()
print("=" * 50)
print("CONSISTENCY CHECK (same image, run twice)")
print(f"  Run 1: {CLASS_NAMES[pt.argmax()]}")
print(f"  Run 2: {CLASS_NAMES[pt2.argmax()]}")
print(f"  Outputs identical: {torch.allclose(pt, pt2)}")
print()
if pt.std() > pr.std() * 2:
    print("VERDICT: Model has LEARNED patterns (high confidence spread vs random)")
else:
    print("VERDICT: Model looks like RANDOM weights (similar spread to untrained)")
