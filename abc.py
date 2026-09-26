
import json
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image


# ------------------------------
# CONFIG (paths)
# ------------------------------
MODEL_PATH = r"D:\Disease Model\models\leaf_disease_resnet50_best.pth"
JSON_PATH  = r"D:\Disease Model\class_names.json"
TEST_IMAGE = r"C:\Users\HP\OneDrive\Documents\Pictures\Screenshots\Screenshot 2026-01-19 113806.png"

DEVICE = torch.device("cpu")  # change to "cuda" if needed

# ------------------------------
# LOAD CLASS NAMES
# ------------------------------
with open(JSON_PATH, "r") as f:
    class_names = json.load(f)

if isinstance(class_names, dict):
    class_names = [class_names[str(i)] for i in range(len(class_names))]

NUM_CLASSES = len(class_names)

# ------------------------------
# LOAD RESNET-50 MODEL (CORRECT)
# ------------------------------
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()

# ------------------------------
# PREPROCESS + GRABCUT (segmentation hta di)
# ------------------------------
def preprocess_leaf(img_path):
    img = Image.open(img_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    tensor = transform(img).unsqueeze(0)
    return tensor.to(DEVICE)

# ------------------------------
# TOP-6 PREDICTION
# ------------------------------
def predict_top6(img_path, topk=6):
    tensor = preprocess_leaf(img_path)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)

        top_probs, top_idxs = torch.topk(
            probs, k=min(topk, NUM_CLASSES)
        )

    print("\nTop Predictions:")
    results = []

    for prob, idx in zip(top_probs[0], top_idxs[0]):
        label = class_names[int(idx)]
        confidence = float(prob) * 100
        results.append((label, confidence))
        print(f"{label:<30} -> {confidence:.2f}%")

    return results

# ------------------------------
# RUN
# ------------------------------
if __name__ == "__main__":
    predict_top6(TEST_IMAGE)
