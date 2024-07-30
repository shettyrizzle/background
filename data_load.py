import tensorflow as tf
import numpy as np
import os

# Paths to dataset directories
train_images_dir = "D:/bgremoval/data/coco/train2017_person_only/"
train_masks_dir = "D:/bgremoval/data/coco/train2017_person_only_masks/"
val_images_dir = "D:/bgremoval/data/coco/val2017_person_only/"
val_masks_dir = "D:/bgremoval/data/coco/val2017_person_only_masks/"

IMG_HEIGHT = 256
IMG_WIDTH = 256

def load_image_mask(image_path, mask_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.convert_image_dtype(mask, tf.float32)
    mask = tf.image.resize(mask, [IMG_HEIGHT, IMG_WIDTH])
    
    return image, mask

def data_augmentation(image, mask):
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)
    
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_up_down(image)
        mask = tf.image.flip_up_down(mask)
    
    if tf.random.uniform(()) > 0.5:
        image = tf.image.rot90(image)
        mask = tf.image.rot90(mask)
    
    return image, mask

def load_dataset(image_dir, mask_dir, batch_size):
    image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir)]
    mask_paths = [os.path.join(mask_dir, fname) for fname in os.listdir(mask_dir)]
    
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    dataset = dataset.shuffle(buffer_size=len(image_paths))
    dataset = dataset.map(lambda img, msk: load_image_mask(img, msk), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(lambda img, msk: data_augmentation(img, msk), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    return dataset

batch_size = 4
train_dataset = load_dataset(train_images_dir, train_masks_dir, batch_size)
val_dataset = load_dataset(val_images_dir, val_masks_dir, batch_size)

# Save metadata (if necessary)
metadata = {
    "batch_size": batch_size
}

np.save('metadata.npy', metadata)

# No need to save datasets to disk, they will be created on-the-fly in training script
