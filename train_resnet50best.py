import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. CONFIGURATION
# ==========================================
BASE_DIR = r"D:\Disease Model\Grabcut segmented"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR   = os.path.join(BASE_DIR, "val")

SAVE_DIR = r"D:\Disease Model\models"
os.makedirs(SAVE_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(SAVE_DIR, "leaf_disease_resnet50_best.pth")
LAST_MODEL_PATH = os.path.join(SAVE_DIR, "leaf_disease_resnet50_last.pth")

BATCH_SIZE = 16
NUM_CLASSES = 14
EPOCHS = 10
LEARNING_RATE = 0.001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. MAIN FUNCTION
# ==========================================
def main():
    print(f"Using device: {DEVICE}")

    # ==========================================
    # DATA TRANSFORMS
    # ==========================================
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # ==========================================
    # DATASET & DATALOADER
    # ==========================================
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    val_dataset   = datasets.ImageFolder(VAL_DIR, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("Classes:", train_dataset.classes)
    print(f"Train images: {len(train_dataset)} | Val images: {len(val_dataset)}")

    # ==========================================
    # MODEL SETUP (ResNet50)
    # ==========================================
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ==========================================
    # TRAINING LOOP
    # ==========================================
    best_val_acc = 0.0
    train_losses = []
    val_accuracies = []

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]")
        for images, labels in loop:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)

        # ======================================
        # VALIDATION
        # ======================================
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        val_accuracies.append(val_acc)

        print(f"\nEpoch {epoch+1} Summary")
        print(f"Train Loss: {avg_loss:.4f}")
        print(f"Val Accuracy: {val_acc:.2f}%")

        # ======================================
        # SAVE LAST MODEL (EVERY EPOCH)
        # ======================================
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_accuracy": val_acc
        }, LAST_MODEL_PATH)

        # ======================================
        # SAVE BEST MODEL
        # ======================================
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"🔥 Best model saved! Val Acc: {best_val_acc:.2f}%")

    print("\nTraining completed.")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")

    # ==========================================
    # PLOTS
    # ==========================================
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies)
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")

    plt.show()

# ==========================================
# ENTRY POINT
# ==========================================
if __name__ == "__main__":
    main()
