use std::{fs, path::PathBuf};
use image::{GrayImage, DynamicImage, open, Luma};
use butter2d::{butterworth, visualize_filter};

fn get_image_path(relative_path: &str) -> PathBuf {
    let current_dir = std::env::current_dir().unwrap();
    current_dir.join(relative_path)
}

fn main() {
    let img_path = get_image_path("images/Lena.png");
    // Read the image
    let img = open(img_path).expect("Failed to open image");

    // Convert to grayscale
    let gray_img = img.into_luma8();

    // Parameters for Butterworth filter
    let cutoff_frequency_ratio = 0.01; // example value
    let high_pass = true; // example value
    let order = 2.0; // example value
    let squared_butterworth = true; // example value
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

    // Save the filtered image
    let filtered_img_path = get_image_path("images/Lena_filtered.png");
    filtered_img.save(filtered_img_path).expect("Failed to save filtered image");

    // Visualize and save the Butterworth filter
    visualize_filter(&filter);
    // Note: The visualize_filter function saves the filter visualization as "butterworth_filter_visualization.png"
}
