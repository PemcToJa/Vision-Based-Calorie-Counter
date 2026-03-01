import copy
import pathlib
import albumentations as A
import cv2
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from torchvision import models
from torch import optim
from src.preprocessing.Food101.data_preprocessing import mapping_return, splits_return
from src.training.datasets.foodClasificationDataset import NutritionDatasetFoodClassification

base_path = pathlib.Path(__file__).resolve().parent.parent.parent

def create_model():
    model = models.efficientnet_v2_s(weights='IMAGENET1K_V1')

    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, 101)
    )

    return model

def train_model(model, train_loader, val_loader, num_epochs=50):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)
    best_acc = 0.0
    best_model_state = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels.data).item()

        epoch_train_loss = train_loss / len(train_loader.dataset)
        epoch_train_acc = (train_correct / len(train_loader.dataset)) * 100

        model.eval()
        val_loss = 0.0
        val_correct = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)

                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data).item()

        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = (val_correct / len(val_loader.dataset)) * 100

        print(f"Epoch {epoch}: Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%, Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%")

        scheduler.step(epoch_val_acc)

        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            best_model_state = copy.deepcopy(model.state_dict())

            model_save_path = base_path / "src" / "training" / "models" / "Food101_Classifier_Best.pth"
            model_save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(best_model_state, model_save_path)
            print("[New best model saved]")

    model.load_state_dict(best_model_state)
    return model

def prepare_loaders(train_transform, val_transform, batch_size=32):
    mappings = mapping_return()
    train_paths, val_paths, test_paths, train_labels, val_labels, test_labels = splits_return()

    train_dataset = NutritionDatasetFoodClassification(
        paths=train_paths,
        labels=train_labels,
        mapping=mappings,
        transform=train_transform
    )

    val_dataset = NutritionDatasetFoodClassification(
        paths=val_paths,
        labels=val_labels,
        mapping=mappings,
        transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader

if __name__ == "__main__":

    train_transform = A.Compose([
        A.LongestMaxSize(max_size=480),
        A.PadIfNeeded(min_height=480, min_width=480, border_mode=cv2.BORDER_CONSTANT),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.HueSaturationValue(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.LongestMaxSize(max_size=480),
        A.PadIfNeeded(min_height=480, min_width=480, border_mode=cv2.BORDER_CONSTANT),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    train_loader, val_loader = prepare_loaders(train_transform, val_transform, batch_size=32)

    model = create_model()

    trained_model = train_model(model, train_loader, val_loader, num_epochs=100)