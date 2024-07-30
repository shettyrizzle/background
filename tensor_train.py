import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.mixed_precision import set_global_policy
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from data_loader import get_data_paths, get_dataset

# Enable mixed precision training
set_global_policy('mixed_float16')

# Define the U-Net model with VGG16 backbone
def unet_model(input_size=(256, 256, 3)):
    inputs = Input(input_size)
    
    base_model = VGG16(include_top=False, weights='imagenet', input_tensor=inputs)
    base_model.trainable = False  # Freeze the base model
    
    c1 = base_model.get_layer("block1_conv2").output
    c2 = base_model.get_layer("block2_conv2").output
    c3 = base_model.get_layer("block3_conv3").output
    c4 = base_model.get_layer("block4_conv3").output
    c5 = base_model.get_layer("block5_conv3").output

    u6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(c6)

    u7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(c7)

    u8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(c8)

    u9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid', dtype='float32')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

# Paths to your datasets
train_images_path = r"D:\bgremoval\data\coco\filtered_train2017_person_only"
train_masks_path = r"D:\bgremoval\data\coco\filtered_train2017_person_only_masks"
val_images_path = r"D:\bgremoval\data\coco\filtered_val2017_person_only"
val_masks_path = r"D:\bgremoval\data\coco\filtered_val2017_person_only_masks"

# Get train and validation data paths
train_image_files, train_mask_files = get_data_paths(train_images_path, train_masks_path)
val_image_files, val_mask_files = get_data_paths(val_images_path, val_masks_path)

# Create datasets
train_dataset = get_dataset(train_image_files, train_mask_files, batch_size=8)
val_dataset = get_dataset(val_image_files, val_mask_files, batch_size=8)

# Build and compile the model
model = unet_model()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),  # Reduced learning rate
              loss=BinaryCrossentropy(from_logits=True),
              metrics=['accuracy', MeanIoU(num_classes=2)])

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, min_lr=1e-6)
model_checkpoint = ModelCheckpoint('new_unet_filtered.h5', save_best_only=True, monitor='val_loss')

# Train the model
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=20,
    callbacks=[early_stopping, reduce_lr, model_checkpoint]
)