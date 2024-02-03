//! # High-Pass Butterworth Filter Example
//!
//! This example demonstrates how to apply a high-pass Butterworth filter to an image using the `butter2d` crate.
//! It outlines the steps to read an image, convert it to grayscale, apply a high-pass Butterworth filter with specified
//! parameters, and then save the filtered image. This process is useful for emphasizing or isolating high-frequency
//! components in an image, such as edges or fine details, by attenuating lower frequencies.
//!
//! ## Steps Overview:
//! 1. **Grayscale Conversion**: Converts a color image to grayscale because the Butterworth filter is applied to
//!    single-channel images in this example.
//! 2. **Reading and Preparing the Image**: Loads an image from a file, manually converts it to grayscale, and prepares
//!    it for filtering.
//! 3. **Applying the Butterworth Filter**: Specifies parameters for the high-pass Butterworth filter (cutoff frequency,
//!    filter order, etc.) and applies it to the grayscale image.
//! 4. **Saving the Filtered Image**: Saves the filtered image to a new file for review and further use.
//!
//! ## Usage:
//! To run this example, an image file at `images/astronaut_gray.png` relative to the current working directory,
//! and ensure there's a writable directory at `output/` for the filtered image. Adjust the filter parameters as needed
//! to achieve the desired high-pass filtering effect.
//!
//! ## Function Descriptions:
//! - `convert_to_grayscale`: Takes an `RgbImage` and converts it to a `GrayImage` using weighted luminance calculation.
//! - `get_image_path`: Constructs an absolute path to an image file based on a relative path provided.
//! - `main`: Orchestrates the image loading, filtering, and saving processes using specified Butterworth filter parameters.

use std::path::PathBuf;
use image::{GrayImage,open, Luma};
use butter2d::butterworth;
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
    let (filtered_img, _filter) = butterworth(
        &gray_img, 
        cutoff_frequency_ratio, 
        high_pass, 
        order, 
        squared_butterworth, 
        npad
    );
    let filtered_img_path = get_image_path("output/astronaut_gray_filtered_rust.png");
    filtered_img.save(filtered_img_path).expect("Failed to save filtered image");
}
