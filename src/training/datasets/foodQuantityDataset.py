import torch
from torch.utils.data import Dataset
import cv2

class NutritionDatasetQuantityRecognition(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_bgr = cv2.imread(str(row["image_rgb_path"]))
        img_depth_bgr = cv2.imread(str(row["image_depth_path"]))

        if img_bgr is None:
            raise FileNotFoundError(f"Photo not found: {row['image_rgb_path']}")
        elif img_depth_bgr is None:
            raise FileNotFoundError(f"Photo not found: {row['image_depth_path']}")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_depth_rgb = cv2.cvtColor(img_depth_bgr, cv2.COLOR_BGR2RGB)

        if self.transform:
            transformed = self.transform(image=img_rgb, depth_image=img_depth_rgb)
            rgb_tensor = transformed["image"]
            depth_tensor = transformed["depth_image"]
        else:
            rgb_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
            depth_tensor = torch.from_numpy(img_depth_rgb).permute(2, 0, 1).float() / 255.0

        mass_val = float(row['mass'])

        target_tensor = torch.tensor([mass_val], dtype=torch.float32).log1p()

        return (rgb_tensor, depth_tensor), target_tensor