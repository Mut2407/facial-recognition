import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from tqdm import tqdm


def main():
    IMG_SIZE = 224
    BATCH_SIZE = 32
    EPOCHS = 30
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    train_dir = "Data/train"
    val_dir = "Data/test"
    num_classes = len(os.listdir(train_dir))

    print(f"Classes detected: {num_classes} — {os.listdir(train_dir)}")


    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=0)

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")


    model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True)
    in_features = model.head.in_features
    model.head = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes)
    )
    model = model.to(DEVICE)


    for param in model.parameters():
        param.requires_grad = True


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)


    best_acc = 0.0

    for epoch in range(EPOCHS):
        print(f"\n===== Epoch {epoch + 1}/{EPOCHS} =====")
        model.train()
        train_loss, train_correct = 0.0, 0

        for images, labels in tqdm(train_loader, desc=f"Training {epoch + 1}/{EPOCHS}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            outputs = model(images)

            if outputs.ndim > 2:
                outputs = torch.mean(outputs, dim=[2, 3])

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == labels).sum().item()

        train_acc = train_correct / len(train_dataset)
        avg_train_loss = train_loss / len(train_loader)

        # ---------- Validation ----------
        model.eval()
        val_correct, val_loss = 0, 0.0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validating"):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                if outputs.ndim > 2:
                    outputs = torch.mean(outputs, dim=[2, 3])
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_correct += (outputs.argmax(1) == labels).sum().item()

        val_acc = val_correct / len(val_dataset)
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch + 1}/{EPOCHS}: "
              f"Train Loss={avg_train_loss:.4f}, Acc={train_acc:.4f} | "
              f"Val Loss={avg_val_loss:.4f}, Acc={val_acc:.4f}")

        # ---------- Save best ----------
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "swin_best_model.pth")
            print("Saved new best model")


    torch.save(model.state_dict(), "swin_final_model.pth")
    print(f"\nTraining finished. Best Val Acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()
