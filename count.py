import os

TRAIN_DIR = r"C:\Users\HP\OneDrive\Desktop\Disease Model\dataset\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\train"

for cls in sorted(os.listdir(TRAIN_DIR)):
    cls_path = os.path.join(TRAIN_DIR, cls)
    if os.path.isdir(cls_path):
        print(f"{cls}: {len(os.listdir(cls_path))}")
