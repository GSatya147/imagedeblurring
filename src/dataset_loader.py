import torch
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import glob

image_transforms = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to 256x256
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
])

class GoProDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_pairs = []

        # Collect all scene directories
        for scene in sorted(os.listdir(root_dir)):
            blur_dir = os.path.join(root_dir, scene, "blur")
            sharp_dir = os.path.join(root_dir, scene, "sharp")

            if os.path.exists(blur_dir) and os.path.exists(sharp_dir):
                for img_name in sorted(os.listdir(blur_dir)):
                    if img_name in os.listdir(sharp_dir):  # Ensure paired images exist
                        self.image_pairs.append((os.path.join(blur_dir, img_name), os.path.join(sharp_dir, img_name)))

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        blur_path, sharp_path = self.image_pairs[idx]

        blur_image = Image.open(blur_path).convert("RGB")
        sharp_image = Image.open(sharp_path).convert("RGB")

        if self.transform:
            blur_image = self.transform(blur_image)
            sharp_image = self.transform(sharp_image)

        return blur_image, sharp_image


train_dataset = GoProDataset(os.path.join(dataset_path, "train"), transform=image_transforms)
test_dataset = GoProDataset(os.path.join(dataset_path, "test"), transform=image_transforms)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)

print(f"Training Samples: {len(train_dataset)}, Testing Samples: {len(test_dataset)}")
