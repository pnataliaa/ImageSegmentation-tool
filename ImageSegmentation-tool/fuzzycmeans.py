import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from skfuzzy.cluster import cmeans


def segment_image_with_fuzzy_cmeans(image_path, n_clusters=None, m=2.0, error=0.005, max_iter=1000):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Nie można załadować obrazu.")

    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values) / 255.0

    pixel_values_flat = pixel_values.T
    cntr, u, _, _, _, _, _ = cmeans(
        pixel_values_flat, c=n_clusters, m=m, error=error, maxiter=max_iter, init=None
    )

    labels = np.argmax(u, axis=0)

    colors = sns.color_palette("viridis", n_clusters)
    colors = (np.array(colors) * 255).astype(np.uint8)

    segmented_image = colors[labels]
    segmented_image = segmented_image.reshape(image.shape)

    return segmented_image, colors, n_clusters

def odmiana_klas(liczba):
    if liczba == 1:
        return "klasa"
    elif 2 <= liczba <= 4 or (22 <= liczba % 100 <= 24):
        return "klasy"
    else:
        return "klas"


if __name__ == "__main__":
    image_path = "/Users/natalia_pyzara/Desktop/PASCAL/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages/000227.jpg"

    original_image = cv2.imread(image_path)
    if original_image is None:
        raise ValueError("Nie można załadować obrazu.")
    original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    segmented_image, cluster_colors, n_clusters = segment_image_with_fuzzy_cmeans(image_path, n_clusters=4)
    segmented_image_rgb = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)

    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    histogram = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

    cluster_centers_gray = [np.mean(color) for color in cluster_colors]
    cluster_centers_gray = sorted(cluster_centers_gray)

    plt.figure(figsize=(16, 8))

    plt.subplot(1, 3, 1)
    plt.imshow(original_image_rgb)
    plt.axis('off')
    plt.title("Oryginalny obraz", fontweight="bold")

    plt.subplot(1, 3, 2)
    plt.plot(histogram, color='black')
    plt.title("Histogram intensywności", fontweight="bold")
    plt.xlabel("Wartość piksela")
    plt.ylabel("Częstotliwość")
    for center in cluster_centers_gray:
        plt.axvline(x=center, color='red', linestyle='--', label=f'Klasa {int(center)}')
    plt.legend(loc="upper right")

    plt.subplot(1, 3, 3)
    plt.imshow(segmented_image_rgb)
    plt.axis('off')
    plt.title(f"Obraz po segmentacji\n(Liczba klas: {n_clusters})", fontweight="bold")

    plt.tight_layout()
    plt.show()

    # Przygotowanie siatki do wykresu
    fig, axs = plt.subplots(2, 5, figsize=(20, 8))
    axs = axs.flatten()

    # Oryginalny obraz w pierwszym miejscu
    axs[0].imshow(original_image_rgb)
    axs[0].axis('off')
    axs[0].set_title("Oryginalny obraz", fontweight="bold")

    # Generowanie obrazów od 2 do 10 klas
    for i, n_clusters in enumerate(range(2, 11)):
        segmented_image = segment_image_with_fuzzy_cmeans(original_image, n_clusters=n_clusters)
        segmented_image_rgb = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)

        axs[i + 1].imshow(segmented_image_rgb)
        axs[i + 1].axis('off')
        axs[i + 1].set_title(f"{n_clusters} klasy", fontweight="bold")

    plt.tight_layout()
    plt.show()


#def find_optimal_clusters_fuzzy(pixel_values, max_clusters=10, m=2.0, error=0.005, max_iter=1000):
#
#     partition_coefficients = []
#     pixel_values_flat = pixel_values.T
#
#     for n_clusters in range(2, max_clusters + 1):
#         _, u, _, _, _, _, _ = cmeans(
#             pixel_values_flat, c=n_clusters, m=m, error=error, maxiter=max_iter, init=None
#         )
#         pc = np.mean(np.sum(u**2, axis=0))
#         partition_coefficients.append(pc)
#
#     optimal_clusters = np.argmax(partition_coefficients) + 2
#
#     plt.figure(figsize=(8, 4))
#     plt.plot(range(2, max_clusters + 1), partition_coefficients, marker='o')
#     plt.xlabel('Liczba klas')
#     plt.ylabel('Partition Coefficient (PC)')
#     plt.title('Optymalna liczba klas (Partition Coefficient)')
#     plt.show()
#
#     return optimal_clusters
    # if n_clusters is None:
    #     n_clusters = find_optimal_clusters_fuzzy(pixel_values)
    #     print(f"Optymalna liczba klas: {n_clusters}")