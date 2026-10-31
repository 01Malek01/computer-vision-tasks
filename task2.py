import cv2
import matplotlib.pyplot as plt
import numpy as np

IMAGE_PATH = "images.jpg"
CLUSTERS = 3
MAX_ITERATIONS = 100


def fit_kmeans(pixels, k, max_iterations):
    rng = np.random.default_rng(10)
    centers = pixels[rng.choice(len(pixels), k, replace=False)].copy()

    for i in range(max_iterations):
        # assign every pixel to its closest center
        distances = ((pixels[:, None] - centers[None]) ** 2).sum(axis=2)
        labels = distances.argmin(axis=1)

        # move each center to the average of its pixels
        new_centers = centers.copy()
        for j in range(k):
            group = pixels[labels == j]
            if len(group):
                new_centers[j] = group.mean(axis=0)

        if np.allclose(centers, new_centers):
            break
        centers = new_centers

    return centers, labels, i + 1


def main():
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        raise FileNotFoundError(f"Could not load {IMAGE_PATH}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, c = image.shape
    pixels = image.reshape(-1, c).astype(np.float32)

    centers, labels, iterations = fit_kmeans(pixels, CLUSTERS, MAX_ITERATIONS)
    colors = np.clip(centers, 0, 255).astype(np.uint8)
    result = colors[labels].reshape(h, w, c)

    print("Pixels:", len(pixels))
    print("Iterations:", iterations)
    print("Colors (RGB):\n", colors)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[1].imshow(result)
    axes[1].set_title("K-means segmentation")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
