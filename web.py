import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn

# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="Leaf Segmentation & Classification",
    page_icon="🌿",
    layout="wide"
)

# ===================== CUSTOM CSS =====================
st.markdown("""
<style>
body { background-color: #f5f7f6; }

.card {
    background-color: white;
    padding: 1.5rem;
    border-radius: 16px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.08);
    margin-bottom: 1.5rem;
}

.title {
    color: #2f855a;
    font-size: 36px;
    font-weight: 700;
}

.subtitle {
    color: #4a5568;
    font-size: 18px;
}

.pred-box {
    background-color: #edfdf5;
    padding: 0.8rem;
    border-radius: 10px;
    margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# ===================== DEVICE =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===================== LOAD CLASS NAMES =====================
with open("plant_classes.txt") as f:
    plant_classes = f.read().splitlines()

# ===================== IMAGE TRANSFORMS =====================
cls_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ===================== SIMPLE U-NET DEFINITION =====================
# ⚠️ Replace this with YOUR EXACT U-Net if different
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# ===================== LOAD MODELS =====================
@st.cache_resource
def load_models():
    # Segmentation model
    unet = UNet().to(device)
    unet.load_state_dict(
        torch.load("C:\Users\HP\OneDrive\Desktop\Plant_AI\segmentation100.pth", map_location=device)
    )
    unet.eval()

    # Classification model
    plant_model = models.resnet50(pretrained=False)
    plant_model.fc = nn.Linear(
        plant_model.fc.in_features, len(plant_classes)
    )
    plant_model.load_state_dict(
        torch.load("C:\Users\HP\OneDrive\Desktop\Plant_AI\model_epoch_20.pth", map_location=device)
    )
    plant_model.eval()

    return unet, plant_model

unet, plant_model = load_models()

# ===================== SEGMENTATION =====================
def segment_leaf(image):
    img = cv2.resize(image, (256, 256)) / 255.0
    img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float().to(device)

    with torch.no_grad():
        mask = torch.sigmoid(unet(img)).squeeze().cpu().numpy()

    mask = (mask > 0.4).astype(np.uint8) * 255
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    return cv2.bitwise_and(image, image, mask=mask)

# ===================== CLASSIFICATION =====================
def classify_top5(image):
    img = Image.fromarray(image)
    img = cls_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = plant_model(img)
        probs = torch.softmax(outputs, dim=1)

    values, indices = torch.topk(probs, 5)

    return [
        (plant_classes[i], float(values[0][j] * 100))
        for j, i in enumerate(indices[0])
    ]

# ===================== UI =====================
st.markdown("<div class='title'>🌿 Leaf Segmentation & Classification</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>U-Net + ResNet based Plant Identification</div><br>", unsafe_allow_html=True)

left, right = st.columns([1, 1.4])

# ---------- LEFT PANEL ----------
with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📤 Upload Leaf Image")
    uploaded_file = st.file_uploader(
        "Supported formats: JPG, PNG",
        type=["jpg", "png", "jpeg"]
    )
    analyze_btn = st.button("🔍 Analyze Image", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- RIGHT PANEL ----------
with right:
    if uploaded_file and analyze_btn:
        image = np.array(Image.open(uploaded_file).convert("RGB"))

        with st.spinner("Processing image..."):
            segmented = segment_leaf(image)
            predictions = classify_top5(segmented)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🖼 Original vs Segmented")
        c1, c2 = st.columns(2)
        c1.image(image, caption="Original", use_column_width=True)
        c2.image(segmented, caption="Segmented Leaf", use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🌱 Top-5 Plant Predictions")
        for name, prob in predictions:
            st.markdown(
                f"<div class='pred-box'>🌿 <b>{name}</b> — {prob:.2f}%</div>",
                unsafe_allow_html=True
            )
        st.markdown("</div>", unsafe_allow_html=True)
