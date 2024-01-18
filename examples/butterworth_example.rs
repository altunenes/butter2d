use std::{fs, path::PathBuf};
use image::{GrayImage, DynamicImage, open};
use butter2d::butterworth; 

fn get_image_path(relative_path: &str) -> PathBuf {
    let current_dir = std::env::current_dir().unwrap();
    current_dir.join(relative_path)
}

fn main() {
    let img_path = get_image_path("images/Lena.png");
    // Read the image
    let img = open(img_path).expect("Failed to open image");
    let gray_img = img.to_luma8(); // Convert to grayscale
    // Define parameters for the Butterworth filter
    let cutoff_frequency_ratio = 0.5; // Example value, adjust as needed
    let high_pass = true; // true for high pass, false for low pass
    let order = 2.0; // Order of the filter
    let squared_butterworth = true; // Use squared Butterworth filter
    let npad = 1; // Padding size
    // Apply the Butterworth filter
    let filtered_img = butterworth(
        &gray_img, 
        cutoff_frequency_ratio, 
        high_pass, 
        order, 
        squared_butterworth, 
        npad
    );

    // Ensure output directory exists
    let output_dir = get_image_path("output");
    if !output_dir.exists() {
        fs::create_dir_all(&output_dir).expect("Failed to create output directory");
    }

    // Save the output
    let output_path = output_dir.join("filtered_Lena.png");
    filtered_img.save(output_path).expect("Failed to save image");
}