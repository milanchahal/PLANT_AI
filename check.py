from torchvision import datasets

BASE_DIR = r"C:\Users\HP\OneDrive\Desktop\Disease Model\dataset"

TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR   = os.path.join(BASE_DIR, "valid")

train_ds = datasets.ImageFolder(TRAIN_DIR)
print("Train samples:", len(train_ds))
print("Classes:", train_ds.classes)
