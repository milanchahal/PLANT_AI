import cv2
import json
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np

# ------------------------------
# CONFIG (change these paths)
# ------------------------------
model_path = r"D:\Disease Model\models\leaf_disease_resnet18.pth"
json_path  = r"D:\Disease Model\class_names.json"
test_image = r"D:\Disease Model\img test\apple healthy1.jpeg"

# ------------------------------
# LOAD CLASS NAMES
# ------------------------------
with open(json_path, "r") as f:
    class_names = json.load(f)

if isinstance(class_names, dict):
    class_names = [class_names[str(i)] for i in range(len(class_names))]

num_classes = len(class_names)
#print("Loaded classes:", num_classes)


# ------------------------------
# LOAD MODEL
# ------------------------------
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, num_classes)

state = torch.load(model_path, map_location="cpu")
model.load_state_dict(state)
model.eval()


# ------------------------------
# PREPROCESS + GRABCUT SEGMENTATION
# ------------------------------
def preprocess_leaf(img_path):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mask = np.zeros(img_rgb.shape[:2], np.uint8)
    bgModel = np.zeros((1, 65), np.float64)
    fgModel = np.zeros((1, 65), np.float64)

    h, w = img_rgb.shape[:2]
    rect = (10, 10, w - 20, h - 20)

    cv2.grabCut(img_rgb, mask, rect, bgModel, fgModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    segmented = img_rgb * mask2[:, :, np.newaxis]

    segmented = Image.fromarray(segmented)

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    tensor = preprocess(segmented).unsqueeze(0)
    return tensor


# ------------------------------
# TOP-6 PREDICTION FUNCTION
# ------------------------------

def predict_top6(img_path, topk=6):
    tensor = preprocess_leaf(img_path)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)

        top_probs, top_idxs = torch.topk(probs, k=min(topk, num_classes))
        top_probs = top_probs.squeeze(0)
        top_idxs = top_idxs.squeeze(0)

    print("\nTop Predictions (percentage):")
    results = []
    for prob, idx in zip(top_probs, top_idxs):
        label = class_names[idx]
        confidence = float(prob) * 100.0   # convert to %
        results.append((label, confidence))
        print(f"{label:<30} -> {confidence:.2f}%")

    return results

# ------------------------------
# RUN
# ------------------------------
if __name__ == "__main__":
    predict_top6(test_image)
