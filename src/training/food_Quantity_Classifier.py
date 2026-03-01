import copy
import pathlib
import albumentations as A
import cv2
import pandas as pd
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from torchvision import models
from torch import optim
from src.training.datasets.foodQuantityDataset import NutritionDatasetQuantityRecognition

base_path = pathlib.Path(__file__).resolve().parent.parent.parent
classifier_path = base_path / "src" / "training" / "models" / "Food_Classifier.pth"

class FoodMassFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.food_classifier_backbone = models.efficientnet_v2_s(weights=None)
        in_features = self.food_classifier_backbone.classifier[1].in_features
        self.food_classifier_backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, 101)
        )

        self.food_classifier_backbone.load_state_dict(
            torch.load(classifier_path, map_location="cuda")
        )

        self.food_classifier_backbone.classifier = nn.Identity()

        for param in self.food_classifier_backbone.parameters():
            param.requires_grad = False

        self.depth_backbone = models.efficientnet_v2_s(weights='DEFAULT')
        self.depth_backbone.classifier = nn.Identity()

        self.regressor = nn.Sequential(
            nn.Linear(1280 + 1280, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x_rgb, x_depth):
        feat_food_classifier = self.food_classifier_backbone(x_rgb)
        feat_depth = self.depth_backbone(x_depth)
        combined = torch.cat((feat_food_classifier, feat_depth), dim=1)
        return self.regressor(combined)

def train_model(model, train_loader, val_loader, num_epochs=50):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    best_loss = float("inf")
    best_model_state = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        if epoch == 15:
            print("ramie food classifier zostało odmrożone")
            for param in model.food_classifier_backbone.parameters():
                param.requires_grad = True

            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-5
        model.train()
        train_loss = 0.0
        train_mae = 0.0

        for (rgb, depth), labels in train_loader:
            rgb, depth, labels = rgb.to(device), depth.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(rgb, depth)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            with torch.no_grad():
                train_mae += torch.abs(torch.expm1(outputs) - torch.expm1(labels)).sum().item()

        epoch_train_loss = train_loss / len(train_loader.dataset)
        epoch_train_mae = train_mae / (len(train_loader.dataset))

        model.eval()
        val_loss = 0.0
        val_mae = 0.0

        with torch.no_grad():
            for (rgb, depth), labels in val_loader:
                rgb, depth, labels = rgb.to(device), depth.to(device), labels.to(device)
                outputs = model(rgb, depth)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * rgb.size(0)
                val_mae += torch.abs(torch.expm1(outputs) - torch.expm1(labels)).sum().item()

        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_mae = val_mae / (len(val_loader.dataset))

        print(f"Epoch {epoch}: Train Loss: {epoch_train_loss:.4f}, Train MAE: {epoch_train_mae:.2f}, Val Loss: {epoch_val_loss:.4f}, Val MAE: {epoch_val_mae:.2f}")

        scheduler.step(epoch_val_loss)

        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            model_save_path = base_path / "src" / "training" / "models" / "Fusion-CalorieNet.pth"
            torch.save(best_model_state, model_save_path)
            print("[New best model saved]")

    model.load_state_dict(best_model_state)
    return model

def prepare_loaders(base_path, train_transform, test_transform, batch_size=32):
    csv_path = base_path / "data" / "processed" / "metadata" / "metadata_clean.csv"

    full_df = pd.read_csv(csv_path)
    test_df = full_df[full_df['split'] == 'test'].copy()
    temp_train_df = full_df[full_df['split'] == 'train'].copy()

    val_df = temp_train_df.sample(frac=0.1, random_state=42)
    train_df = temp_train_df.drop(val_df.index)

    train_ds = NutritionDatasetQuantityRecognition(train_df, transform=train_transform)
    val_ds = NutritionDatasetQuantityRecognition(val_df, transform=test_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

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
    ], additional_targets={'depth_image': 'image'})

    test_transform = A.Compose([
        A.LongestMaxSize(max_size=480),
        A.PadIfNeeded(min_height=480, min_width=480, border_mode=cv2.BORDER_CONSTANT),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], additional_targets={'depth_image': 'image'})


    train_loader, val_loader = prepare_loaders(base_path, train_transform, test_transform)

    model = FoodMassFusion()

    trained_model = train_model(model, train_loader, val_loader, num_epochs=100)