import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
from tqdm import tqdm

# Define the U-Net model (same architecture as used during training)
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Function to load the model
def load_model(model_path, device):
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# Function to perform background removal
def remove_background(input_folder, output_folder, model, device):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    for image_name in tqdm(os.listdir(input_folder)):
        image_path = os.path.join(input_folder, image_name)
        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            mask = output.squeeze().cpu().numpy()

        mask = (mask > 0.5).astype("uint8") * 255
        mask_image = Image.fromarray(mask).resize(image.size, Image.BILINEAR)
        image = Image.composite(image, Image.new("RGB", image.size, (255, 255, 255)), mask_image)

        output_image_path = os.path.join(output_folder, image_name)
        image.save(output_image_path)

if __name__ == "__main__":
    input_folder = "G:\My Drive\inputimages"
    output_folder = "output_images"
    model_path = "best_unet_model.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)

    remove_background(input_folder, output_folder, model, device)
