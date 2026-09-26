import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np

# ===============================
# CONFIG (edit these)
# ===============================
MODEL_PATH = r"D:\Disease Model\plant_disease_resnet18.pth"
IMAGE_PATH = r"D:\Disease Model\test Leafmold1.jpg"
IMAGE_SIZE = (224, 224)  # input size for your ResNet model

# Class names (must match training)
CLASS_NAMES = [
    "Apple__Apple_scab","Apple__Black_rot","Apple__Cedar_apple_rust","Apple__healthy",
    "Tomato__Bacterial_spot","Tomato__Early_blight","Tomato__healthy","Tomato__Late_blight",
    "Tomato__Leaf_Mold","Tomato__Septoria_leaf_spot","Tomato__Spider_mites Two-spotted_spider_mite",
    "Tomato__Target_Spot","Tomato__Tomato_mosaic_virus","Tomato__Tomato_Yellow_Leaf_Curl_Virus"
]

# ===============================
# LOAD MODEL
# ===============================
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# ===============================
# GRABCUT SEGMENTATION FUNCTION
# ===============================
def grabcut_leaf(img):
    # create mask
    mask = np.zeros(img.shape[:2], np.uint8)

    # define background/foreground models
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # define rectangle around leaf (rough estimate, can be full image)
    h, w = img.shape[:2]
    rect = (5, 5, w-10, h-10)  # slightly inside the borders

    # apply GrabCut
    cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    # convert mask to binary: 0,2 -> background, 1,3 -> foreground
    mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
    img_segmented = img * mask2[:, :, np.newaxis]

    return img_segmented

# ===============================
# IMAGE PREPROCESSING FUNCTION
# ===============================
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    
    # Apply GrabCut segmentation to isolate leaf
    img = grabcut_leaf(img)
    
    # Resize and normalize for ResNet
    img = cv2.resize(img, IMAGE_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # batch dimension

    return img

# ===============================
# PREDICTION FUNCTION
# ===============================
def predict_leaf(img_path, top_k=5):
    img_array = preprocess_image(img_path)
    preds = model.predict(img_array)[0]  # get 1D array

    # Get top-k predictions
    top_indices = preds.argsort()[-top_k:][::-1]
    print(f"Top-{top_k} predictions:")
    for i, idx in enumerate(top_indices):
        print(f"{i+1}. {CLASS_NAMES[idx]} - {preds[idx]*100:.2f}%")

# ===============================
# RUN PREDICTION
# ===============================
predict_leaf(IMAGE_PATH)
