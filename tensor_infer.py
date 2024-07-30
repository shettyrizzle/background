import os
import argparse
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image

def remove_background(image, mask):
    # Ensure mask is binary
    mask = (mask > 0.5).astype(np.uint8)
    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

    # Apply mask to the image
    background_removed = image * mask
    
    # Replace black areas with transparency
    transparent_image = np.dstack((background_removed, mask[:, :, 0] * 255))

    return transparent_image

def process_folder(input_folder, output_folder, model_path, target_size):
    model = load_model(model_path, compile=False)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '.png')

            image = load_img(input_path, target_size=target_size)
            image_array = img_to_array(image) / 255.0
            image_array = np.expand_dims(image_array, axis=0)

            # Predict mask
            mask = model.predict(image_array)[0, :, :, 0]

            # Remove background
            original_image = img_to_array(load_img(input_path))
            mask_resized = np.array(Image.fromarray((mask * 255).astype(np.uint8)).resize(original_image.shape[1::-1], Image.BILINEAR)) / 255.0
            output_image = remove_background(original_image, mask_resized)

            # Save result as PNG
            output_image_pil = Image.fromarray(output_image.astype(np.uint8))
            output_image_pil.save(output_path, format='PNG')

            print(f"Processed: {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Remove background from images in a folder.')
    parser.add_argument('--input_folder', type=str, required=True, help='Path to the input folder containing images.')
    parser.add_argument('--output_folder', type=str, required=True, help='Path to the output folder to save processed images.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model file.')
    parser.add_argument('--target_size', type=int, nargs=2, default=(256, 256), help='Target size for image resizing.')

    args = parser.parse_args()

    process_folder(args.input_folder, args.output_folder, args.model_path, tuple(args.target_size))
