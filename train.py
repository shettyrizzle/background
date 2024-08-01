import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models.segmentation as segmentation
from data import create_data_loaders  # Ensure correct import path

def train_model(model, dataloader, criterion, optimizer, device, num_epochs=24):
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch in dataloader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device).unsqueeze(1)  # Add channel dimension to mask
            
            optimizer.zero_grad()
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / (len(dataloader)):.4f}')

def main():
    image_dir = 'D:\\bgremoval\\data\\coco\\train2017_person_only'
    mask_dir = 'D:\\bgremoval\\data\\coco\\train2017_person_only_masks'
    batch_size = 16
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_loader = create_data_loaders(image_dir, mask_dir, batch_size)
    
    # Load and adjust the model
    model = segmentation.fcn_resnet50(weights='FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1')
    model.classifier[4] = nn.Conv2d(512, 1, kernel_size=1)  # Adjust final layer for binary segmentatio
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    train_model(model, train_loader, criterion, optimizer, device)

if __name__ == '__main__':
    main()
