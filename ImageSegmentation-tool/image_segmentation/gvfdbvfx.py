import os
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
import pandas as pd

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

def calculate_class_metrics(true_mask, predicted_mask, n_classes):
    class_metrics = {}
    for class_idx in range(n_classes):
        true_binary = (true_mask == class_idx).flatten()
        pred_binary = (predicted_mask == class_idx).flatten()

        if np.sum(true_binary) > 0 or np.sum(pred_binary) > 0:
            precision = precision_score(true_binary, pred_binary, zero_division=0)
            recall = recall_score(true_binary, pred_binary, zero_division=0)
            f1 = f1_score(true_binary, pred_binary, zero_division=0)
            iou = jaccard_score(true_binary, pred_binary, zero_division=0)

            intersection = np.sum((true_mask == class_idx) & (predicted_mask == class_idx))
            union = np.sum(true_mask == class_idx) + np.sum(predicted_mask == class_idx)
            dice = 2 * intersection / (union + 1e-6)

            true_negative = np.sum((true_mask != class_idx) & (predicted_mask != class_idx))
            false_positive = np.sum((true_mask != class_idx) & (predicted_mask == class_idx))
            specificity = true_negative / (true_negative + false_positive + 1e-6)

            total_pixels = len(true_binary)
            correct_pixels = np.sum(true_binary == pred_binary)
            accuracy = correct_pixels / total_pixels

            class_metrics[class_idx] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'iou': iou,
                'dice': dice,
                'specificity': specificity,
                'accuracy': accuracy
            }

    return class_metrics


# Wczytywanie danych testowych
print("Loading test data...")
images_dir = os.path.join(PATCHES_DIR, "images")
masks_dir = os.path.join(PATCHES_DIR, "masks")
X_test, y_test = load_test_data(images_dir, masks_dir)

# Wykonywanie predykcji i obliczanie metryk
all_metrics = []
print("Performing predictions...")
for idx in range(len(X_test)):
    test_img = X_test[idx]
    test_mask = y_test[idx][:, :, 0]
    img_path = os.path.join(images_dir, sorted(os.listdir(images_dir))[idx])
    mask_path = os.path.join(masks_dir, sorted(os.listdir(masks_dir))[idx])

    prediction = model.predict(np.expand_dims(test_img, axis=0))
    predicted_mask = np.argmax(prediction[0], axis=-1)

    metrics_per_class = calculate_class_metrics(test_mask, predicted_mask, N_CLASSES)
    if metrics_per_class:
        image_metrics = {metric: np.mean([metrics[metric] for metrics in metrics_per_class.values()])
                         for metric in ['precision', 'recall', 'f1', 'iou', 'dice', 'specificity', 'accuracy']}
        image_metrics['image_path'] = img_path
        image_metrics['mask_path'] = mask_path
        all_metrics.append(image_metrics)

    print(f"Metrics for Image {idx + 1}:")
    for metric, value in image_metrics.items():
        if metric not in ['image_path', 'mask_path']:
            print(f"  {metric}: {value:.4f}")

overall_mean_metrics = {metric: np.mean([m[metric] for m in all_metrics if metric in m])
                        for metric in ['precision', 'recall', 'f1', 'iou', 'dice', 'specificity', 'accuracy']}

df_results = pd.DataFrame(all_metrics)
overall_row = {'image_path': 'Overall', 'mask_path': 'Overall'}
overall_row.update(overall_mean_metrics)
df_results = pd.concat([df_results, pd.DataFrame([overall_row])], ignore_index=True)

output_file = "segmentation_results.csv"
df_results.to_csv(output_file, index=False)

print(f"Results saved to {output_file}")
