import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from patchify import patchify
from PIL import Image
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
import json

DATASET_DIR = "satellite_dataset"
PATCH_SIZE = 256
IMG_CHANNELS = 3
N_CLASSES = 6

CLASS_COLORS = {
    "Building": [60, 16, 152],
    "Land": [132, 41, 246],
    "Road": [110, 193, 228],
    "Vegetation": [254, 221, 58],
    "Water": [226, 169, 41],
    "Unlabeled": [155, 155, 155],
}
CLASS_MAPPING = {tuple(color): idx for idx, color in enumerate(CLASS_COLORS.values())}

def rgb_to_2D_label(mask):
    label_seg = np.zeros(mask.shape[:2], dtype=np.uint8)
    for rgb, idx in CLASS_MAPPING.items():
        label_seg[np.all(mask == rgb, axis=-1)] = idx
    return label_seg

def preprocess_tiles(tile_names, root_dir, patch_size):
    image_dataset, mask_dataset = [], []

    for tile_name in tile_names:
        tile_path = os.path.join(root_dir, tile_name)
        image_folder = os.path.join(tile_path, "images")
        mask_folder = os.path.join(tile_path, "masks")

        for image_name in sorted(os.listdir(image_folder)):
            if image_name.endswith(".jpg"):
                image_path = os.path.join(image_folder, image_name)
                image = cv2.imread(image_path)
                SIZE_X = (image.shape[1] // patch_size) * patch_size
                SIZE_Y = (image.shape[0] // patch_size) * patch_size
                image = Image.fromarray(image).crop((0, 0, SIZE_X, SIZE_Y))
                patches_img = patchify(np.array(image), (patch_size, patch_size, 3), step=patch_size)
                for i in range(patches_img.shape[0]):
                    for j in range(patches_img.shape[1]):
                        single_patch_img = patches_img[i, j, 0] / 255.0
                        image_dataset.append(single_patch_img)

        for mask_name in sorted(os.listdir(mask_folder)):
            if mask_name.endswith(".png"):
                mask_path = os.path.join(mask_folder, mask_name)
                mask = cv2.imread(mask_path)
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
                SIZE_X = (mask.shape[1] // patch_size) * patch_size
                SIZE_Y = (mask.shape[0] // patch_size) * patch_size
                mask = Image.fromarray(mask).crop((0, 0, SIZE_X, SIZE_Y))
                patches_mask = patchify(np.array(mask), (patch_size, patch_size, 3), step=patch_size)
                for i in range(patches_mask.shape[0]):
                    for j in range(patches_mask.shape[1]):
                        single_patch_mask = patches_mask[i, j, 0]
                        single_patch_mask = rgb_to_2D_label(single_patch_mask)
                        mask_dataset.append(single_patch_mask)

    return np.array(image_dataset), np.expand_dims(np.array(mask_dataset), axis=-1)

TRAIN_TILES = ["Tile 1", "Tile 2", "Tile 3", "Tile 4", "Tile 5", "Tile 6"]
TEST_TILES = ["Tile 7", "Tile 8"]

print("Preprocessing training Tiles...")
X_train, y_train = preprocess_tiles(TRAIN_TILES, DATASET_DIR, PATCH_SIZE)

print("Preprocessing testing Tiles...")
X_test, y_test = preprocess_tiles(TEST_TILES, DATASET_DIR, PATCH_SIZE)

y_train_cat = to_categorical(y_train, num_classes=N_CLASSES)
y_test_cat = to_categorical(y_test, num_classes=N_CLASSES)

def multi_unet_model(n_classes, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
    inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = inputs

    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = tf.keras.layers.Dropout(0.2)(c1)
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = tf.keras.layers.Dropout(0.2)(c2)
    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = tf.keras.layers.Dropout(0.2)(c3)
    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = tf.keras.layers.Dropout(0.2)(c4)
    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = tf.keras.layers.Dropout(0.3)(c5)
    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    merge6 = tf.keras.layers.concatenate([u6, c4])
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(merge6)
    c6 = tf.keras.layers.Dropout(0.2)(c6)
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    merge7 = tf.keras.layers.concatenate([u7, c3])
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(merge7)
    c7 = tf.keras.layers.Dropout(0.2)(c7)
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    merge8 = tf.keras.layers.concatenate([u8, c2])
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(merge8)
    c8 = tf.keras.layers.Dropout(0.2)(c8)
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    merge9 = tf.keras.layers.concatenate([u9, c1])
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(merge9)
    c9 = tf.keras.layers.Dropout(0.2)(c9)
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    outputs = tf.keras.layers.Conv2D(n_classes, (1, 1), activation='softmax')(c9)
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    return model


model = multi_unet_model(n_classes=N_CLASSES, IMG_HEIGHT=PATCH_SIZE, IMG_WIDTH=PATCH_SIZE, IMG_CHANNELS=IMG_CHANNELS)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

EPOCHS = 50
BATCH_SIZE = 16
history = model.fit(X_train, y_train_cat, validation_data=(X_test, y_test_cat), batch_size=BATCH_SIZE, epochs=EPOCHS)


# model.save("unet_model_satellite.keras")

# print("Performing predictions on test images...")
# test_img_number = np.random.randint(len(X_test))
# test_img = X_test[test_img_number]
# test_mask = np.argmax(y_test[test_img_number], axis=-1)
# prediction = model.predict(np.expand_dims(test_img, axis=0))
# predicted_mask = np.argmax(prediction[0], axis=-1)
#
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 3, 1)
# plt.title("Image")
# plt.imshow(test_img)
# plt.subplot(1, 3, 2)
# plt.title("True Mask")
# plt.imshow(test_mask)
# plt.subplot(1, 3, 3)
# plt.title("Predicted Mask")
# plt.imshow(predicted_mask)
# plt.show()


model_path = "unet_model_satellite.keras"

plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Dokładność treningu', marker='o')
plt.plot(history.history['val_accuracy'], label='Dokładność walidacji', marker='o')
plt.title('Dokładność modelu w kolejnych epokach', fontsize=16, weight='bold')
plt.xlabel('Epoki', fontsize=14, weight='bold')
plt.ylabel('Dokładność', fontsize=14, weight='bold')
plt.legend(fontsize=12)
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Strata treningu', marker='o')
plt.plot(history.history['val_loss'], label='Strata walidacji', marker='o')
plt.title('Strata modelu w kolejnych epokach', fontsize=16, weight='bold')
plt.xlabel('Epoki', fontsize=14, weight='bold')
plt.ylabel('Strata', fontsize=14, weight='bold')
plt.legend(fontsize=12)
plt.grid(True)
plt.show()

model = load_model(model_path)

history_path = "training_history.json"
with open(history_path, "r") as f:
    history = json.load(f)

print(history.keys())