import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def load_image_grayscale(path):
    """Load an image in grayscale."""
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def visualize_images(img1, img2, title1='Image 1', title2='Image 2', output_path='output/visual_comparison.png'):
    """Visualize two images side by side for comparison and save the figure."""
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img1, cmap='gray')
    plt.title(title1)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(img2, cmap='gray')
    plt.title(title2)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(output_path)  # Save the figure
    plt.show()

def visualize_difference(img1, img2, output_path='output/difference.png'):
    """Visualize the difference between two images and save the figure."""
    difference = cv2.absdiff(img1, img2)
    plt.figure(figsize=(5, 5))
    plt.imshow(difference, cmap='hot')
    plt.title('Difference')
    plt.axis('off')
    plt.colorbar()
    plt.savefig(output_path)  # Save the figure
    plt.show()

    mse = np.mean((img1.astype("float") - img2.astype("float")) ** 2)
    print(f"Mean Squared Error: {mse}")

rust_filtered_img_path = "output/astronaut_gray_filtered_rust.png"
python_filtered_img_path = "output/astronaut_gray_filtered_python.png"

# Ensure the output directory exists
os.makedirs('output', exist_ok=True)

rust_img = load_image_grayscale(rust_filtered_img_path)
python_img = load_image_grayscale(python_filtered_img_path)

# Specify paths for saving the comparison and difference images
visual_comparison_path = 'output/visual_comparison.png'
difference_path = 'output/difference_visualization.png'

visualize_images(rust_img, python_img, 'Rust Filtered Image', 'Python (skimage) Filtered Image', visual_comparison_path)
visualize_difference(rust_img, python_img, difference_path)