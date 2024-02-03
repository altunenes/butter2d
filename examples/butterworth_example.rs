use std::{fs, path::PathBuf};
use image::{GrayImage, DynamicImage, open, Luma};
use butter2d::{butterworth, visualize_filter};
use image::{Pixel, RgbImage};
fn convert_to_grayscale(img: &RgbImage) -> GrayImage {
    let mut gray_img = GrayImage::new(img.width(), img.height());

    for (x, y, pixel) in img.enumerate_pixels() {
        let rgb = pixel.to_rgb();
        // Use the same weights as OpenCV for grayscale conversion
        let luma = (0.299 * rgb[0] as f64 + 0.587 * rgb[1] as f64 + 0.114 * rgb[2] as f64) as u8;
        gray_img.put_pixel(x, y, Luma([luma]));
    }

    gray_img
}
fn get_image_path(relative_path: &str) -> PathBuf {
    let current_dir = std::env::current_dir().unwrap();
    current_dir.join(relative_path)
}
fn main() {
    let img_path = get_image_path("images/astronaut_gray.png");
    // Read the image
    let img = open(&img_path).expect("Failed to open image").to_rgb8();
    
    // Manually convert to grayscale
    let gray_img = convert_to_grayscale(&img);
    // Parameters for Butterworth filter
    let cutoff_frequency_ratio: f64 = 0.1; // example value
    let high_pass = true; // example value
    let order = 2.0; // example value
    let squared_butterworth = false; // example value
    let npad = 0; // example value for padding
    // Apply Butterworth filter
    let (filtered_img, filter) = butterworth(
        &gray_img, 
        cutoff_frequency_ratio, 
        high_pass, 
        order, 
        squared_butterworth, 
        npad
    );
    let filtered_img_path = get_image_path("C:/Users/enes-/OneDrive/Masaüstü/pyt/Lena_filtered_rust3.png");
    filtered_img.save(filtered_img_path).expect("Failed to save filtered image");
    // Visualize and save the Butterworth filter
    visualize_filter(&filter);
}
