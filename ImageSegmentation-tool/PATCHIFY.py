from patchify import patchify
from PIL import Image
from sklearn.metrics import jaccard_score, precision_score, recall_score, f1_score
import numpy as np
import os
from tqdm import tqdm
import cv2

DATASET_DIR = "satellite_dataset"
PATCH_SIZE = 256
TILES = ["Tile 7", "Tile 8"]
OUTPUT_DIR = "patches_output"

def save_patches(tiles, root_dir, patch_size, output_dir):
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)

    for tile_name in tiles:
        print(f"Processing {tile_name}...")
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
                        patch = patches_img[i, j, 0]
                        patch_filename = f"{tile_name}_{image_name.split('.')[0]}_patch_{i}_{j}.jpg"
                        patch_path = os.path.join(output_dir, "images", patch_filename)
                        patch_img = Image.fromarray(patch)
                        patch_img.save(patch_path)

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
                        patch = patches_mask[i, j, 0]
                        patch_filename = f"{tile_name}_{mask_name.split('.')[0]}_patch_{i}_{j}.png"
                        patch_path = os.path.join(output_dir, "masks", patch_filename)
                        patch_img = Image.fromarray(patch)
                        patch_img.save(patch_path)

    print(f"All patches saved in {output_dir}")

# save_patches(TILES, DATASET_DIR, PATCH_SIZE, OUTPUT_DIR)


def calculate_unet_metrics(predicted_folder, ground_truth_folder, patch_size):
    metrics = {
        "iou_micro": [],
        "precision_micro": [],
        "recall_micro": [],
        "f1_micro": [],
        "accuracy": [],
        "dice_micro": [],
        "specificity_micro": [],
    }

    predicted_files = sorted(os.listdir(predicted_folder))
    ground_truth_files = sorted(os.listdir(ground_truth_folder))

    if len(predicted_files) != len(ground_truth_files):
        raise ValueError("Mismatch between number of predicted and ground truth patches.")

    for pred_name, gt_name in tqdm(zip(predicted_files, ground_truth_files), total=len(predicted_files)):
        if not pred_name.endswith(".png") or not gt_name.endswith(".png"):
            continue

        pred_path = os.path.join(predicted_folder, pred_name)
        gt_path = os.path.join(ground_truth_folder, gt_name)

        pred_mask = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        pred_flat = pred_mask.flatten()
        gt_flat = gt_mask.flatten()

        metrics["iou_micro"].append(jaccard_score(gt_flat, pred_flat, average='micro'))
        metrics["precision_micro"].append(precision_score(gt_flat, pred_flat, average='micro', zero_division=0))
        metrics["recall_micro"].append(recall_score(gt_flat, pred_flat, average='micro', zero_division=0))
        metrics["f1_micro"].append(f1_score(gt_flat, pred_flat, average='micro', zero_division=0))

        intersection = np.sum((gt_mask == pred_mask) & (gt_mask > 0))
        union = np.sum(gt_mask > 0) + np.sum(pred_mask > 0)
        dice_score = 2 * intersection / (union + 1e-6)
        metrics["dice_micro"].append(dice_score)

        true_negative = np.sum((gt_mask == 0) & (pred_mask == 0))
        false_positive = np.sum((gt_mask == 0) & (pred_mask > 0))
        specificity = true_negative / (true_negative + false_positive + 1e-6)
        metrics["specificity_micro"].append(specificity)

        correct_pixels = np.sum(gt_mask == pred_mask)
        total_pixels = gt_mask.size
        metrics["accuracy"].append(correct_pixels / total_pixels)

    final_metrics = {key: np.mean(values) for key, values in metrics.items()}
    return final_metrics

metrics = calculate_unet_metrics(os.path.join(OUTPUT_DIR, "masks"),
                                  os.path.join(OUTPUT_DIR, "masks"),
                                  PATCH_SIZE)

print("Calculated Metrics:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")