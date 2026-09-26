import zipfile
import os

ZIP_PATH = "archive.zip"
OUTPUT_DIR = "dataset"

TARGET_CLASSES = ["Apple___", "Tomato___"]
SPLITS = ["train", "valid"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
    for file in zip_ref.namelist():
        for split in SPLITS:
            for cls in TARGET_CLASSES:
                if f"/{split}/{cls}" in file.replace("\\", "/"):
                    print("Extracting:", file)
                    zip_ref.extract(file, OUTPUT_DIR)

print("✅ Extraction finished")
