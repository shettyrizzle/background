import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import albumentations as A
import albumentations.pytorch as AP

class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_filenames = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
        
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        image_path = os.path.join(self.image_dir, img_name)
        mask_name = img_name.replace('.jpg', '_mask.png')  # Adjust according to your naming convention
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # Convert mask to grayscale
        
        # Convert images to numpy arrays
        image = np.array(image)
        mask = np.array(mask)
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        return {'image': image, 'mask': mask}

def get_transforms():
    return A.Compose([
        A.RandomResizedCrop(height=256, width=256),
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        AP.ToTensorV2()
    ])

def create_data_loaders(image_dir, mask_dir, batch_size):
    dataset = CustomDataset(image_dir, mask_dir, transform=get_transforms())
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

