import torch
from torch.utils.data import Dataset
import cv2

class NutritionDatasetFoodClassification(Dataset):
    def __init__(self, paths, labels, mapping, transform=None):
        self.paths = paths
        self.labels = labels
        self.mapping = mapping
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        img_bgr = cv2.imread(str(img_path))

        if img_bgr is None:
            raise FileNotFoundError(f"Photo not found: {img_path}")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        if self.transform:
            transformed = self.transform(image=img_rgb)
            rgb_tensor = transformed["image"]
        else:
            rgb_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0

        label_name = self.labels[idx]
        label_idx = self.mapping[label_name]
        label_tensor = torch.tensor(label_idx, dtype=torch.long)

        return rgb_tensor, label_tensor