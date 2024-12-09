import os
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score

model = tf.keras.models.load_model("unet_model_satellite.keras")

PATCHES_DIR = "patches_output"
IMG_CHANNELS = 3
N_CLASSES = 6

CLASS_COLORS = {
    0: [60, 16, 152],
    1: [132, 41, 246],
    2: [110, 193, 228],
    3: [254, 221, 58],
    4: [226, 169, 41],
    5: [155, 155, 155],
}


def rgb_to_2D_label(mask):
    label_seg = np.zeros(mask.shape[:2], dtype=np.uint8)
    for rgb, idx in {tuple(v): k for k, v in CLASS_COLORS.items()}.items():
        label_seg[np.all(mask == rgb, axis=-1)] = idx
    return label_seg


def decode_segmentation(mask, class_colors):
    decoded_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_idx, color in class_colors.items():
        decoded_mask[mask == class_idx] = color
    return decoded_mask


def load_test_data(images_dir, masks_dir):
    image_dataset, mask_dataset = [], []

    for img_name in sorted(os.listdir(images_dir)):
        if img_name.endswith(".jpg"):
            img_path = os.path.join(images_dir, img_name)
            img = np.array(Image.open(img_path)) / 255.0
            image_dataset.append(img)

    for mask_name in sorted(os.listdir(masks_dir)):
        if mask_name.endswith(".png"):
            mask_path = os.path.join(masks_dir, mask_name)
            mask = np.array(Image.open(mask_path))
            mask = rgb_to_2D_label(mask)
            mask_dataset.append(mask)

    return np.array(image_dataset), np.expand_dims(np.array(mask_dataset), axis=-1)

def calculate_unet_metrics(true_mask, predicted_mask):
    metrics = {}

    true_flat = true_mask.flatten()
    pred_flat = predicted_mask.flatten()

    class_counts = np.bincount(true_flat, minlength=N_CLASSES)
    total_pixels = np.sum(class_counts)
    class_weights = class_counts / total_pixels

    metrics['iou_weighted'] = jaccard_score(true_flat, pred_flat, average='weighted')

    metrics['precision_weighted'] = precision_score(true_flat, pred_flat, average='weighted', zero_division=0)
    metrics['recall_weighted'] = recall_score(true_flat, pred_flat, average='weighted', zero_division=0)
    metrics['f1_weighted'] = f1_score(true_flat, pred_flat, average='weighted', zero_division=0)

    dice_per_class = []
    for c in range(N_CLASSES):
        intersection = np.sum((true_flat == c) & (pred_flat == c))
        union = np.sum(true_flat == c) + np.sum(pred_flat == c)
        if union > 0:
            dice_per_class.append((2 * intersection / union) * class_weights[c])
    metrics['dice_weighted'] = np.sum(dice_per_class)


    specificity_per_class = []
    for c in range(N_CLASSES):
        true_neg = np.sum((true_flat != c) & (pred_flat != c))
        false_pos = np.sum((true_flat != c) & (pred_flat == c))
        specificity = true_neg / (true_neg + false_pos + 1e-6)
        specificity_per_class.append(specificity * class_weights[c])
    metrics['specificity_weighted'] = np.sum(specificity_per_class)

    correct_pixels = np.sum(true_flat == pred_flat)
    metrics['accuracy_weighted'] = correct_pixels / total_pixels

    return metrics


def get_metrics_for_unet(image_path, metrics_df):

    image_name = os.path.basename(image_path)

    filtered_df = metrics_df[metrics_df['segmentation_metrics_results.csv'].str.contains(image_name, na=False)]

    if filtered_df.empty:
        return None

    metrics_row = filtered_df.iloc[0]

    metrics = {
        "iou": metrics_row["iou_micro"],
        "precision": metrics_row["precision_micro"],
        "recall": metrics_row["recall_micro"],
        "f1": metrics_row["f1_micro"],
        "dice": metrics_row["dice_micro"],
        "specificity": metrics_row["specificity_micro"],
        "accuracy": metrics_row["accuracy"],
    }
    return metrics



print("Loading test data...")
images_dir = os.path.join(PATCHES_DIR, "images")
masks_dir = os.path.join(PATCHES_DIR, "masks")
X_test, y_test = load_test_data(images_dir, masks_dir)

all_metrics = []
print("Performing predictions...")

for idx in range(len(X_test)):
    test_img = X_test[idx]
    test_mask = y_test[idx][:, :, 0]

    prediction = model.predict(np.expand_dims(test_img, axis=0))
    predicted_mask = np.argmax(prediction[0], axis=-1)

    metrics = calculate_unet_metrics(test_mask, predicted_mask)
    all_metrics.append(metrics)

    print(f"Metrics for Image {idx + 1}:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")


import pandas as pd

results = []
for idx, metrics in enumerate(all_metrics):
    input_file_path = os.path.join(images_dir, sorted(os.listdir(images_dir))[idx])
    mask_file_path = os.path.join(masks_dir, sorted(os.listdir(masks_dir))[idx])

    result = {
        'ID': idx + 1,
        'Input Image': input_file_path,
        'Ground Truth Mask': mask_file_path,
    }
    result.update(metrics)
    results.append(result)

df_results = pd.DataFrame(results)

output_file = "segmentation_weighted_metrics_results.csv"
df_results.to_csv(output_file, index=False)
print(f"Results saved to {output_file}")


