import numpy as np
import cv2
import matplotlib.pyplot as plt


def multithreshold_colored(img, thresholds, colormap='viridis'):
    segments = np.zeros_like(img, dtype=np.uint8)
    thresholds = sorted(thresholds)
    for i, t in enumerate(thresholds):
        if i == 0:
            segments[img <= t] = i
        else:
            segments[(img > thresholds[i - 1]) & (img <= t)] = i
    segments[img > thresholds[-1]] = len(thresholds)

    num_segments = len(thresholds) + 1
    cmap = plt.get_cmap(colormap)
    colors = cmap(np.linspace(0, 1, num_segments))[:, :3]
    segmented_colored = np.zeros((*img.shape, 3), dtype=np.float32)
    for i in range(num_segments):
        segmented_colored[segments == i] = colors[i]

    return (segmented_colored * 255).astype(np.uint8)


def otsu_method(hist):
    num_bins = hist.shape[0]
    total = hist.sum()
    sum_total = np.dot(range(0, num_bins), hist)

    weight_background = 0.0
    sum_background = 0.0

    optimum_value = 0
    maximum = -np.inf

    for t in range(0, num_bins):
        weight_background += hist.item(t)
        if weight_background == 0:
            continue
        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break

        sum_background += t * hist.item(t)
        mean_foreground = (sum_total - sum_background) / weight_foreground
        mean_background = sum_background / weight_background

        inter_class_variance = weight_background * weight_foreground * \
            (mean_background - mean_foreground) ** 2

        if inter_class_variance > maximum:
            optimum_value = t
            maximum = inter_class_variance
    return optimum_value, maximum


def normalised_histogram_binning(hist, M=32, L=256):
    norm_hist = np.zeros((M, 1), dtype=np.float32)
    N = L // M
    counters = [range(x, x + N) for x in range(0, L, N)]
    for i, C in enumerate(counters):
        norm_hist[i] = 0
        for j in C:
            norm_hist[i] += hist[j]
    norm_hist = (norm_hist / norm_hist.max()) * 100
    return norm_hist


def find_valleys(H):
    hsize = H.shape[0]
    probs = np.zeros((hsize, 1), dtype=int)
    for i in range(1, hsize - 1):
        if H[i] < H[i - 1] and H[i] < H[i + 1]:
            probs[i] = 1
    valleys = [i for i, x in enumerate(probs) if x > 0]
    return valleys


def valley_estimation(hist, M=32, L=256):
    norm_hist = normalised_histogram_binning(hist, M, L)
    valleys = find_valleys(norm_hist)
    return valleys


def threshold_valley_regions(hist, valleys, N):
    thresholds = []
    for valley in valleys:
        start_pos = (valley * N) - N
        end_pos = (valley + 2) * N
        h = hist[start_pos:end_pos]
        sub_threshold, val = otsu_method(h)
        thresholds.append((start_pos + sub_threshold, val))
    thresholds.sort(key=lambda x: x[1], reverse=True)
    thresholds, values = [list(t) for t in zip(*thresholds)]
    return thresholds


def modified_TSMO(hist, M=32, L=256):
    N = L // M
    valleys = valley_estimation(hist, M, L)
    thresholds = threshold_valley_regions(hist, valleys, N)
    return thresholds

def apply_modified_otsu_with_multithreshold(image):
    histogram, bins = np.histogram(image.ravel(), bins=256, range=(0, 256))
    thresholds = modified_TSMO(histogram, M=32, L=256)
    segmented_colored = multithreshold_colored(image, thresholds)

    if len(segmented_colored.shape) == 2:
        segmented_colored = cv2.cvtColor(segmented_colored, cv2.COLOR_GRAY2RGB)

    return segmented_colored


if __name__ == "__main__":
    image_path = "dataset_image_segmentation/train/10_jpg.rf.68504d449dd333632ea80d4a945b0a77.jpg"
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("Nie udało się wczytać obrazu. Sprawdź ścieżkę.")
    else:
        histogram, bins = np.histogram(image.ravel(), bins=256, range=(0, 256))

        suggested_thresholds = modified_TSMO(histogram, M=32, L=256)
        suggested_classes = len(suggested_thresholds) + 1

        segmented_colored = multithreshold_colored(image, suggested_thresholds, colormap='viridis')

        fig, axs = plt.subplots(2, 5, figsize=(18, 8))
        axs = axs.flatten()

        axs[0].imshow(image, cmap="gray")
        axs[0].axis("off")
        axs[0].set_title(r"$\bf{Oryginalny\ obraz}$")

        for i, num_classes in enumerate(range(2, 11)):
            idx = i + 1
            if num_classes == suggested_classes:
                thresholds = suggested_thresholds
            else:
                thresholds = np.linspace(0, 255, num_classes + 1, dtype=int)[1:-1]

            segmented_temp = multithreshold_colored(image, thresholds, colormap='viridis')
            axs[idx].imshow(segmented_temp)
            axs[idx].axis("off")
            if num_classes == suggested_classes:
                axs[idx].set_title(rf"$\bf{{{num_classes}\ klasy\ (sugerowane)}}$")
            else:
                axs[idx].set_title(rf"$\bf{{{num_classes}\ klasy}}$")

        for j in range(len(range(2, 11)) + 1, len(axs)):
            fig.delaxes(axs[j])

        plt.suptitle(r"$\bf{Segmentacja\ z\ różną\ liczbą\ klas}$", fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        plt.show()

        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        axs[0].imshow(image, cmap="gray")
        axs[0].set_title(r"$\bf{Oryginalny\ obraz}$")
        axs[0].axis("off")

        axs[1].plot(bins[:-1], histogram, color='black')
        for t in suggested_thresholds:
            axs[1].axvline(t, color='red', linestyle="--", label=f"Próg: {t}")
        axs[1].set_title(r"$\bf{Histogram\ z\ progami}$")
        axs[1].set_xlabel("Intensywność")
        axs[1].set_ylabel("Liczba pikseli")
        axs[1].legend()

        axs[2].imshow(segmented_colored)
        axs[2].set_title(r"$\bf{Obraz\ po\ segmentacji}$")
        axs[2].axis("off")

        plt.tight_layout()
        plt.show()

        unique_classes = np.unique(segmented_colored)
        print(f"Liczba klas w obrazie po segmentacji: {len(suggested_thresholds) + 1}")

# import matplotlib
# import matplotlib.pyplot as plt
# import numpy as np
# from skimage import data
# from skimage.filters import threshold_multiotsu
# from scipy.signal import find_peaks
# import cv2
#
# matplotlib.rcParams['font.size'] = 9
#
# image = data.camera()
#
#
# small_image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2), interpolation=cv2.INTER_LINEAR)
#
# histogram, bin_edges = np.histogram(small_image.ravel(), bins=128, range=(0, 256))
#
# peaks, _ = find_peaks(histogram, distance=20)
# num_classes = len(peaks) + 1
# print(f"Detected number of classes (from peaks): {num_classes}")
#
# optimal_classes = 2
# max_variance = 0
#
# for classes in range(2, 10):
#     thresholds = threshold_multiotsu(small_image, classes=classes)
#     regions = np.digitize(small_image, bins=thresholds)
#     variance = np.var(regions)
#     if variance > max_variance:
#         max_variance = variance
#         optimal_classes = classes
#
# print(f"Optymalna liczba klas: {optimal_classes}")
#
# thresholds = threshold_multiotsu(small_image, classes=optimal_classes)
#
# regions = np.digitize(image, bins=thresholds)
#
# fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
#
# ax[0].imshow(image, cmap='gray')
# ax[0].set_title('Original')
# ax[0].axis('off')
#
# smoothed_histogram = np.convolve(histogram, np.ones(5) / 5, mode='same')
# ax[1].plot(bin_edges[:-1], smoothed_histogram, color='black')
# ax[1].set_title('Histogram with Peaks and Thresholds')
# ax[1].set_xlabel('Pixel Intensity')
# ax[1].set_ylabel('Frequency')
#
# for peak in peaks:
#     ax[1].plot(bin_edges[peak], smoothed_histogram[peak], "x", color="blue", label="Peaks")
# for thresh in thresholds:
#     ax[1].axvline(thresh, color='red', linestyle='--', label=f'Threshold: {thresh:.2f}')
#
# ax[1].legend()
#
# ax[2].imshow(regions, cmap='jet')
# ax[2].set_title('Multi-Otsu Result')
# ax[2].axis('off')
#
# plt.tight_layout()
# plt.show()

