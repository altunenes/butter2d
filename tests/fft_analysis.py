import numpy as np
import matplotlib.pyplot as plt
import cv2
def compute_fft(image):
    """Compute the FFT of an image."""
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.log(np.abs(fshift) + 1) 
    phase_spectrum = np.angle(fshift)
    return magnitude_spectrum, phase_spectrum

def compare_spectrums(rust_img, python_img, output_path):
    """Compare the magnitude and phase spectrums of two images."""
    rust_magnitude, rust_phase = compute_fft(rust_img)
    python_magnitude, python_phase = compute_fft(python_img)
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs[0, 0].imshow(rust_magnitude, cmap='gray')
    axs[0, 0].set_title('Rust Magnitude Spectrum')
    axs[0, 0].axis('off')
    axs[0, 1].imshow(python_magnitude, cmap='gray')
    axs[0, 1].set_title('Python Magnitude Spectrum')
    axs[0, 1].axis('off')
    axs[1, 0].imshow(rust_phase, cmap='gray')
    axs[1, 0].set_title('Rust Phase Spectrum')
    axs[1, 0].axis('off')
    axs[1, 1].imshow(python_phase, cmap='gray')
    axs[1, 1].set_title('Python Phase Spectrum')
    axs[1, 1].axis('off')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()
rust_img_path = "output/astronaut_gray_filtered_rust.png"
python_img_path = "output/astronaut_gray_filtered_python.png"
rust_img = cv2.imread(rust_img_path, cv2.IMREAD_GRAYSCALE)
python_img = cv2.imread(python_img_path, cv2.IMREAD_GRAYSCALE)
output_path = "output/spectrum_comparison.png"
compare_spectrums(rust_img, python_img, output_path)
