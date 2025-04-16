use butter2d::{butterworth, pad_image, get_nd_butterworth_filter, fft2d};
use image::{GrayImage, Luma};
use ndarray::{Array, Array2, IxDyn};
use rustfft::num_complex::Complex;
use nalgebra::DMatrix;

fn generate_test_image(width: u32, height: u32) -> GrayImage {
    let mut img = GrayImage::new(width, height);
    for (x, _, pixel) in img.enumerate_pixels_mut() {
        let intensity = ((x as f64 / width as f64) * 255.0) as u8;
        *pixel = Luma([intensity]);
    }
    img
}

#[test]
fn test_butterworth_2d_zeros() {
    let img = GrayImage::new(4, 4);
    let (filtered_img, _) = butterworth(&img, 0.5, true, 2.0, true, 0);
    for y in 0..filtered_img.height() {
        for x in 0..filtered_img.width() {
            assert_eq!(*filtered_img.get_pixel(x, y), Luma([0u8]));
        }
    }
}

#[test]
fn test_butterworth_cutoff_parameters() {
    let cutoffs = [0.2, 0.3];
    let orders = [6.0, 10.0];
    let high_passes = [false, true];
    let squared_options = [false, true];
    for &cutoff in &cutoffs {
        for &order in &orders {
            for &high_pass in &high_passes {
                for &squared_butterworth in &squared_options {
                    println!("Test parameters: cutoff={}, order={}, high_pass={}, squared_butterworth={}", 
                        cutoff, order, high_pass, squared_butterworth);
                }
            }
        }
    }
}

#[test]
fn test_butterworth_cutoff() {
    // Define parameters for testing
    let cutoffs = [0.2, 0.3];
    let orders = [6.0, 10.0];
    let high_passes = [false, true];
    let squared_options = [false, true];
    for &cutoff in &cutoffs {
        for &order in &orders {
            for &high_pass in &high_passes {
                for &squared_butterworth in &squared_options {
                    let img = GrayImage::from_pixel(128, 128, Luma([128])); // Reduced size for faster tests
                    let (filtered_img, _) = butterworth(&img, cutoff, high_pass, order, squared_butterworth, 0);
                    let mut changed = false;
                    for y in 0..filtered_img.height() {
                        for x in 0..filtered_img.width() {
                            if *filtered_img.get_pixel(x, y) != Luma([128]) {
                                changed = true;
                                break;
                            }
                        }
                        if changed { break; }
                    }
                    assert!(changed, "Filter should modify the image.");
                }
            }
        }
    }
}

// Updated fft_magnitude function
fn fft_magnitude(image: &GrayImage) -> Array2<f64> {
    let nrows = image.height() as usize;
    let ncols = image.width() as usize;
    let mut data = Vec::with_capacity(nrows * ncols);
    for p in image.pixels() {
        data.push(Complex::new(p.0[0] as f64, 0.0));
    }
    let dmatrix = DMatrix::from_vec(nrows, ncols, data);
    
    // Use the library's fft2d function
    let fft_result = fft2d(&dmatrix);
    
    let magnitude_array = Array::from_shape_fn((nrows, ncols), |(row, col)| {
        fft_result[(row, col)].norm()
    });
    
    magnitude_array
}

#[test]
fn test_butterworth_frequency_response2() {
    // Create a synthetic test image with different frequency components
    let mut img = GrayImage::new(256, 256);
    let center_x = 128.0;
    let center_y = 128.0;
    
    // Fill the image with a constant mid-gray value first
    for y in 0..256 {
        for x in 0..256 {
            img.put_pixel(x, y, Luma([128]));
        }
    }
    
    // Add a circle with soft edges (low frequency)
    for y in 0..256 {
        for x in 0..256 {
            let dist = ((x as f64 - center_x).powi(2) + (y as f64 - center_y).powi(2)).sqrt();
            if dist < 100.0 {
                // Gradually fade from center to edge
                let intensity = 200.0 - (dist / 100.0) * 150.0;
                img.put_pixel(x, y, Luma([intensity as u8]));
            }
        }
    }
    
    // Add some sharp edges (high frequency components)
    for y in 30..70 {
        for x in 30..70 {
            img.put_pixel(x, y, Luma([250]));
        }
    }
    
    for y in 180..220 {
        for x in 180..220 {
            img.put_pixel(x, y, Luma([50]));
        }
    }
    
    // Apply a high-pass Butterworth filter
    let cutoff_frequency_ratio = 0.15;
    let order = 2.0;
    let high_pass = true;
    let squared_butterworth = true;
    
    let (filtered_img, _) = butterworth(&img, cutoff_frequency_ratio, high_pass, order, squared_butterworth, 0);
    
    // Count how many pixels are far from mid-gray in the filtered image
    // High-pass filters should set smooth areas to mid-gray (128) and preserve edges
    let mut smooth_area_count = 0;
    let mut significant_pixels = 0;
    
    for y in 0..256 {
        for x in 0..256 {
            let dist = ((x as f64 - center_x).powi(2) + (y as f64 - center_y).powi(2)).sqrt();
            let is_smooth_area = dist < 50.0 || (x > 80 && x < 170 && y > 80 && y < 170);
            
            let filtered_val = filtered_img.get_pixel(x, y).0[0] as i32;
            let diff_from_mid = (filtered_val - 128).abs();
            
            if diff_from_mid > 20 {
                significant_pixels += 1;
                
                // Count how many significant pixels are in smooth areas
                if is_smooth_area {
                    smooth_area_count += 1;
                }
            }
        }
    }
    
    // Calculate the ratio of significant pixels in non-smooth vs. smooth areas
    let smooth_area_percentage = smooth_area_count as f64 / significant_pixels as f64 * 100.0;
    
    println!("High-pass filter test:");
    println!("  Total significant pixels: {}", significant_pixels);
    println!("  Smooth area significant pixels: {}", smooth_area_count);
    println!("  Percentage in smooth areas: {:.2}%", smooth_area_percentage);
    
    // A good high-pass filter should retain very few significant pixels in smooth areas
    assert!(smooth_area_percentage < 40.0, 
        "High-pass filter should primarily highlight edges, not smooth areas");
}

#[test]
fn test_butterworth_2d_realfft_comparison() {
    let img = generate_test_image(256, 256); // Reduced size for faster tests
    let cutoff_frequency_ratio = 0.2;
    let order = 2.0;
    let high_pass = false;
    let squared_butterworth = true;
    let (filtered_real_img, _) = butterworth(&img, cutoff_frequency_ratio, high_pass, order, squared_butterworth, 0);
    let real_img_magnitude_spectrum = fft_magnitude(&filtered_real_img);
    let avg_magnitude = real_img_magnitude_spectrum.iter().sum::<f64>() / real_img_magnitude_spectrum.len() as f64;
    println!("Average magnitude of the frequency response: {}", avg_magnitude);
}

#[test]
fn test_butterworth_frequency_response() {
    // Create a test image with high frequency details
    let mut img = GrayImage::new(256, 256);
    
    // Create a checker pattern (high frequency content)
    for y in 0..256 {
        for x in 0..256 {
            let color = if (x / 8 + y / 8) % 2 == 0 { 255 } else { 0 };
            img.put_pixel(x, y, Luma([color]));
        }
    }
    
    // Apply a low-pass filter (should blur the checkerboard)
    let cutoff_frequency_ratio = 0.1; // Low cutoff = more aggressive smoothing
    let order = 2.0;
    let high_pass = false; // Low-pass filter
    let squared_butterworth = false;
    
    let (filtered_img, _) = butterworth(&img, cutoff_frequency_ratio, high_pass, order, squared_butterworth, 0);
    
    // Calculate variation in original vs filtered image
    // Original checkerboard has high variation, filtered should have less
    let mut orig_variation = 0.0;
    let mut filtered_variation = 0.0;
    
    for y in 1..255 {
        for x in 1..255 {
            // Calculate local variation (difference from neighbors)
            let center_orig = img.get_pixel(x, y).0[0] as f64;
            let neighbors_orig = [
                img.get_pixel(x-1, y).0[0] as f64,
                img.get_pixel(x+1, y).0[0] as f64,
                img.get_pixel(x, y-1).0[0] as f64,
                img.get_pixel(x, y+1).0[0] as f64,
            ];
            let local_var_orig = neighbors_orig.iter()
                .map(|&n| (n - center_orig).powi(2))
                .sum::<f64>() / 4.0;
            
            let center_filtered = filtered_img.get_pixel(x, y).0[0] as f64;
            let neighbors_filtered = [
                filtered_img.get_pixel(x-1, y).0[0] as f64,
                filtered_img.get_pixel(x+1, y).0[0] as f64,
                filtered_img.get_pixel(x, y-1).0[0] as f64,
                filtered_img.get_pixel(x, y+1).0[0] as f64,
            ];
            let local_var_filtered = neighbors_filtered.iter()
                .map(|&n| (n - center_filtered).powi(2))
                .sum::<f64>() / 4.0;
            
            orig_variation += local_var_orig;
            filtered_variation += local_var_filtered;
        }
    }
    
    // Normalize
    orig_variation /= (254 * 254) as f64;
    filtered_variation /= (254 * 254) as f64;
    
    println!("Original image variation: {}", orig_variation);
    println!("Filtered image variation: {}", filtered_variation);
    
    // For a low-pass filter, the filtered image should have less variation
    assert!(filtered_variation < orig_variation * 0.8, 
            "Low-pass filter should significantly reduce high-frequency variation");
}

#[test]
fn test_padding_correctness() {
    let img = generate_test_image(10, 10);
    let npad = 5;
    let padded_img = pad_image(&img, npad);
    assert_eq!(padded_img.nrows(), (10 + 2 * npad).next_power_of_two());
    assert_eq!(padded_img.ncols(), (10 + 2 * npad).next_power_of_two());
}

#[test]
fn test_butterworth_filter_generation() {
    let shape = [64, 64];
    let cutoff_frequency_ratio = 0.1;
    let order = 2.0;
    let high_pass = true;
    let squared_butterworth = true;
    let filter = get_nd_butterworth_filter(
        &shape,
        cutoff_frequency_ratio,
        order,
        high_pass,
        true,
        squared_butterworth,
    );
    assert_eq!(filter.dim(), IxDyn(&shape), "Filter dimensions mismatch");
    let cutoff_index = (cutoff_frequency_ratio * shape[0] as f64).round() as usize;
    let cutoff_value = filter[[cutoff_index, cutoff_index]].re;
    if high_pass {
        assert!(cutoff_value > 0.0, "High-pass filter cutoff response incorrect");
    } else {
        assert!(cutoff_value < 1.0, "Low-pass filter cutoff response incorrect");
    }
}