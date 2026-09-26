import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import os

# ================================
# CONFIG
# ================================
MODEL_PATH = "plant_disease_resnet18.pth"
CLASS_NAMES_PATH = "class_names.json"
IMAGE_PATH = r"C:\Users\HP\OneDrive\Desktop\Disease Model\bacterial_spot_tomato.jpg"   # 🔴 CHANGE THIS
TOP_K = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# ================================
# LOAD CLASS NAMES
# ================================
with open(CLASS_NAMES_PATH, "r") as f:
    classes = json.load(f)

NUM_CLASSES = len(classes)

# ================================
# MODEL
# ================================
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

print("✅ Model loaded successfully")

# ================================
# TRANSFORMS (same as validation)
# ================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ================================
# LOAD IMAGE
# ================================
img = Image.open(IMAGE_PATH).convert("RGB")
img = transform(img).unsqueeze(0).to(DEVICE)

# ================================
# INFERENCE
# ================================
with torch.no_grad():
    outputs = model(img)
    probs = torch.softmax(outputs, dim=1)

    top_probs, top_idxs = torch.topk(probs, TOP_K)

print("\n🔍 Top-5 Predictions:\n")

for i in range(TOP_K):
    cls_name = classes[top_idxs[0][i].item()]
    confidence = top_probs[0][i].item() * 100
    print(f"{i+1}. {cls_name} → {confidence:.2f}%")
