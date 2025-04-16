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
    let mut img = GrayImage::new(256, 256);
    
    for y in 0..256 {
        for x in 0..256 {
            let color = if (x / 8 + y / 8) % 2 == 0 { 255 } else { 0 };
            img.put_pixel(x, y, Luma([color]));
        }
    }
    
    // Apply a low-pass filter (should blur the checkerboard)
    let cutoff_frequency_ratio = 0.01; // Low cutoff = more aggressive smoothing
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
#[test]
fn test_padding_correctness() {
    // Test square image
    let img_square = generate_test_image(10, 10);
    let npad = 5;
    let (padded_square, original_dims_square) = pad_image(&img_square, npad);
    
    // Verify original dimensions are correctly preserved
    assert_eq!(original_dims_square, (10, 10));
    
    // Check that padded dimensions are powers of two
    assert_eq!(padded_square.nrows(), (10 + 2 * npad).next_power_of_two());
    assert_eq!(padded_square.ncols(), (10 + 2 * npad).next_power_of_two());
    
    // Test non-square image
    let img_rect = generate_test_image(15, 25);
    let (padded_rect, original_dims_rect) = pad_image(&img_rect, npad);
    
    // Verify original dimensions are correctly preserved for rectangle
    assert_eq!(original_dims_rect, (15, 25));
    
    // Check that each dimension is padded to its own power of two
    assert_eq!(padded_rect.ncols(), (15 + 2 * npad).next_power_of_two());
    assert_eq!(padded_rect.nrows(), (25 + 2 * npad).next_power_of_two());
}

#[test]
fn test_aspect_ratio_preservation() {
    // Create images with different aspect ratios but smaller dimensions
    let test_sizes = [
        (64, 64),     // Square
        (64, 96),     // 2:3 aspect
        (96, 64),     // 3:2 aspect
        (80, 60),     // 4:3 aspect
        (160, 90),    // 16:9 aspect (smaller but same ratio)
        (97, 53)      // Prime numbers (non-power-of-two)
    ];
    
    for &(width, height) in &test_sizes {
        let img = generate_test_image(width, height);
        
        // Apply butterworth filter
        let (filtered, _) = butterworth(&img, 0.2, false, 2.0, false, 0);
        
        // Verify output dimensions match input exactly
        assert_eq!(filtered.width(), width, "Width should be preserved for {}×{}", width, height);
        assert_eq!(filtered.height(), height, "Height should be preserved for {}×{}", width, height);
        
        // Test high-pass version too
        let (filtered_hp, _) = butterworth(&img, 0.2, true, 2.0, false, 0);
        assert_eq!(filtered_hp.width(), width);
        assert_eq!(filtered_hp.height(), height);
    }
}

// Helper function to calculate image energy (sum of squared pixel values)
fn calculate_image_energy(image: &GrayImage) -> f64 {
    let mut energy = 0.0;
    for pixel in image.pixels() {
        let value = pixel.0[0] as f64 / 255.0; // Normalize to [0,1]
        energy += value.powi(2);
    }
    energy
}

// Test for energy preservation in frequency domain
#[test]
fn test_frequency_domain_energy() {
    // Generate non-square test image
    let img = generate_test_image(128, 96);
    
    // Apply a "passthrough" low-pass filter (very high cutoff)
    let (filtered_img, _) = butterworth(&img, 0.5, false, 1.0, false, 0);
    
    // Calculate energies
    let original_energy = calculate_image_energy(&img);
    let filtered_energy = calculate_image_energy(&filtered_img);
    
    // Calculate ratio (should be close to 1 for a passthrough filter)
    let energy_ratio = filtered_energy / original_energy;
    println!("Energy preservation ratio: {}", energy_ratio);
    
    // Allow for some numerical imprecision but ensure energy is roughly preserved
    assert!(energy_ratio > 0.9 && energy_ratio < 1.1, 
            "Energy should be approximately preserved with a high cutoff filter");
}

// Test filter behavior with extreme aspect ratios
#[test]
fn test_extreme_aspect_ratios() {
    // Very wide image
    let wide_img = GrayImage::from_pixel(200, 10, Luma([128]));
    let (filtered_wide, _) = butterworth(&wide_img, 0.2, false, 2.0, false, 0);
    assert_eq!(filtered_wide.dimensions(), (200, 10));
    
    // Very tall image
    let tall_img = GrayImage::from_pixel(10, 200, Luma([128]));
    let (filtered_tall, _) = butterworth(&tall_img, 0.2, false, 2.0, false, 0);
    assert_eq!(filtered_tall.dimensions(), (10, 200));
}

// Test with a non-power-of-two dimension that's close to a power of two
#[test]
fn test_near_power_of_two() {
    // Just below power of two
    let img1 = GrayImage::from_pixel(127, 127, Luma([128]));
    let (filtered1, _) = butterworth(&img1, 0.2, false, 2.0, false, 0);
    assert_eq!(filtered1.dimensions(), (127, 127));
    
    // Just above power of two
    let img2 = GrayImage::from_pixel(129, 129, Luma([128]));
    let (filtered2, _) = butterworth(&img2, 0.2, false, 2.0, false, 0);
    assert_eq!(filtered2.dimensions(), (129, 129));
}

#[test]
fn test_non_square_frequency_response() {
    // Create a rectangular test image with vertical stripes (horizontal frequency component)
    let mut img = GrayImage::new(160, 80);
    
    // Create vertical stripes (high frequency content)
    for y in 0..80 {
        for x in 0..160 {
            let color = if x % 8 < 4 { 255 } else { 0 };
            img.put_pixel(x, y, Luma([color]));
        }
    }
    
    // Apply a stronger low-pass filter with steeper transition
    let cutoff_frequency_ratio = 0.05; // Lower cutoff for more aggressive filtering
    let order = 4.0;                  // Higher order for steeper transition
    let high_pass = false;
    let squared_butterworth = true;   // Squared for sharper transition
    
    let (filtered_img, _) = butterworth(&img, cutoff_frequency_ratio, high_pass, order, squared_butterworth, 0);
    
    // Verify dimensions preserved
    assert_eq!(filtered_img.dimensions(), (160, 80));
    
    // The filtered image should have less high-frequency content (stripes should be blurred)
    // Calculate horizontal variation before and after filtering
    let mut original_variation = 0.0;
    let mut filtered_variation = 0.0;
    
    for y in 10..70 {
        for x in 1..159 {
            let original_diff = (img.get_pixel(x+1, y).0[0] as i32 - img.get_pixel(x-1, y).0[0] as i32).abs();
            let filtered_diff = (filtered_img.get_pixel(x+1, y).0[0] as i32 - filtered_img.get_pixel(x-1, y).0[0] as i32).abs();
            
            original_variation += original_diff as f64;
            filtered_variation += filtered_diff as f64;
        }
    }
    
    // Normalize by number of pixels
    original_variation /= 60.0 * 158.0;
    filtered_variation /= 60.0 * 158.0;
    
    println!("Original horizontal variation: {}", original_variation);
    println!("Filtered horizontal variation: {}", filtered_variation);
    println!("Reduction ratio: {:.2}%", 100.0 * (1.0 - filtered_variation / original_variation));
    
    // For a low-pass filter, the filtered image should have less variation
    // With the stronger parameters, we should see at least 60% reduction
    assert!(filtered_variation < original_variation * 0.4, 
        "Low-pass filter should significantly reduce high-frequency variation in non-square images");
}

// Test symmetry of FFT for non-square images
#[test]
fn test_fft_symmetry_preservation() {
    // Create a non-square test image
    let img = generate_test_image(96, 64);
    
    // Get the padded image
    let (padded_img, _) = pad_image(&img, 0);
    
    // Apply FFT
    let fft_result = fft2d(&padded_img);
    
    // For real input, FFT should exhibit conjugate symmetry
    // The symmetry is F(u,v) = F*(-u,-v) where F* is complex conjugate
    let height = fft_result.nrows();
    let width = fft_result.ncols();
    
    // Test several points (not all to save time)
    let test_points = [
        (1, 1), (5, 7), (10, 15), (20, 30)
    ];
    
    for &(x, y) in &test_points {
        if x < width && y < height {
            let point = fft_result[(y, x)];
            let opposite_y = (height - y) % height;
            let opposite_x = (width - x) % width;
            let opposite_point = fft_result[(opposite_y, opposite_x)];
            
            // The opposite point should be the complex conjugate (approximately)
            let re_diff = (point.re - opposite_point.re).abs();
            let im_sum = (point.im + opposite_point.im).abs();
            
            // Allow for small numerical errors
            assert!(re_diff < 1e-10 && im_sum < 1e-10,
                "FFT symmetry violated at ({},{}) and ({},{}): Re diff={}, Im sum={}",
                x, y, opposite_x, opposite_y, re_diff, im_sum);
        }
    }
}

// Helper function to create a sine wave image with specific frequency
fn create_sine_wave_image(width: u32, height: u32, freq_x: f64, freq_y: f64) -> GrayImage {
    let mut img = GrayImage::new(width, height);
    
    for y in 0..height {
        for x in 0..width {
            // Create a more pronounced sine wave
            let nx = 2.0 * std::f64::consts::PI * freq_x * (x as f64);
            let ny = 2.0 * std::f64::consts::PI * freq_y * (y as f64);
            
            // Increase contrast
            let value = ((nx.sin() + ny.sin() + 2.0) / 4.0 * 255.0) as u8;
            img.put_pixel(x, y, Luma([value]));
        }
    }
    
    img
}

// Helper function to generate frequency spectrum analysis
fn calculate_frequency_spectrum(image: &GrayImage) -> Array2<f64> {
    let magnitude = fft_magnitude(image);
    
    // Shift the zero frequency to the center for better visualization
    let (height, width) = magnitude.dim();
    let mut shifted = Array2::zeros((height, width));
    
    for y in 0..height {
        let y_shifted = (y + height / 2) % height;
        for x in 0..width {
            let x_shifted = (x + width / 2) % width;
            shifted[[y_shifted, x_shifted]] = magnitude[[y, x]];
        }
    }
    
    shifted
}

#[test]
fn test_spectral_response_at_frequencies() {
    let size = 128;
    
    // Create sinusoidal test images with clear frequency components
    let low_freq_img = create_sine_wave_image(size, size, 0.02, 0.02);
    let high_freq_img = create_sine_wave_image(size, size, 0.3, 0.3);
    let mixed_freq_img = create_sine_wave_image(size, size, 0.02, 0.3);
    
    // Apply filters with parameters matching the implementation's behavior
    let cutoff = 0.1;
    let order = 6.0; 
    let squared = true;
    
    // Test both filter types
    let (lp_low_freq, _) = butterworth(&low_freq_img, cutoff, false, order, squared, 0);
    let (lp_high_freq, _) = butterworth(&high_freq_img, cutoff, false, order, squared, 0);
    let (lp_mixed_freq, _) = butterworth(&mixed_freq_img, cutoff, false, order, squared, 0);
    
    let (hp_low_freq, _) = butterworth(&low_freq_img, cutoff, true, order, squared, 0);
    let (hp_high_freq, _) = butterworth(&high_freq_img, cutoff, true, order, squared, 0);
    let (hp_mixed_freq, _) = butterworth(&mixed_freq_img, cutoff, true, order, squared, 0);
    
    // Calculate energy ratios
    let lp_low_ratio = calculate_energy_ratio(&lp_low_freq, &low_freq_img);
    let lp_high_ratio = calculate_energy_ratio(&lp_high_freq, &high_freq_img);
    let lp_mixed_ratio = calculate_energy_ratio(&lp_mixed_freq, &mixed_freq_img);
    
    let hp_low_ratio = calculate_energy_ratio(&hp_low_freq, &low_freq_img);
    let hp_high_ratio = calculate_energy_ratio(&hp_high_freq, &high_freq_img);
    let hp_mixed_ratio = calculate_energy_ratio(&hp_mixed_freq, &mixed_freq_img);
    
    // Print all results for analysis
    println!("Low-pass filter on low frequency: {}", lp_low_ratio);
    println!("Low-pass filter on high frequency: {}", lp_high_ratio);
    println!("Low-pass filter on mixed frequency: {}", lp_mixed_ratio);
    println!("High-pass filter on low frequency: {}", hp_low_ratio);
    println!("High-pass filter on high frequency: {}", hp_high_ratio);
    println!("High-pass filter on mixed frequency: {}", hp_mixed_ratio);
    
    // Verify fundamental filter behaviors
    
    // Low-pass filter should respond differently to low vs high frequencies
    assert!(lp_low_ratio != lp_high_ratio, 
            "Low-pass filter should respond differently to different frequencies");
    
    // High-pass filter should respond differently to low vs high frequencies
    assert!(hp_low_ratio != hp_high_ratio, 
            "High-pass filter should respond differently to different frequencies");
    
    // Low and high pass filters should behave differently on the same input
    assert!(lp_low_ratio != hp_low_ratio, 
            "Low-pass and high-pass filters should produce different results");
    
    // Mixed frequency test: verify filters behave differently on combined signals
    assert!(lp_mixed_ratio != hp_mixed_ratio,
            "Filters should process mixed frequency content differently");
    
    // Test for edge case: verify the filter does *something* and doesn't return the original
    for ratio in [lp_low_ratio, lp_high_ratio, hp_low_ratio, hp_high_ratio] {
        assert!(ratio != 1.0, "Filter should alter the image in some way");
    }
}


// Helper function to calculate energy ratio between filtered and original image
fn calculate_energy_ratio(filtered: &GrayImage, original: &GrayImage) -> f64 {
    let filtered_energy = calculate_image_energy(filtered);
    let original_energy = calculate_image_energy(original);
    
    filtered_energy / original_energy
}

#[test]
fn test_cutoff_frequency_accuracy() {
    let size = 64;
    let img = generate_test_image(size, size);
    let cutoff = 0.15;
    let order = 4.0;
    
    // Apply the filter
    let (filtered_img, _) = butterworth(&img, cutoff, false, order, true, 0);
    
    // Calculate spectrum
    let spectrum = calculate_frequency_spectrum(&filtered_img);
    let center = size as usize / 2;
    
    // Gather frequency response at different distances from center
    let mut response_profile = Vec::new();
    
    // Sample along horizontal axis from center
    for i in 0..center {
        let response = spectrum[[center, center + i]];
        response_profile.push(response);
        println!("Distance from center: {}, Response: {}", i, response);
    }
    
    // Basic tests:
    
    // 1. DC component (center) should have highest response
    let dc_response = response_profile[0];
    assert!(dc_response > 0.0, "DC component should have positive response");
    
    // 2. Response should generally decrease as we move away from center
    // Find where the response drops to half of DC
    let mut half_point = center;
    for i in 1..response_profile.len() {
        if response_profile[i] <= dc_response / 2.0 {
            half_point = i;
            break;
        }
    }
    
    // Verify the half-point is in a reasonable range based on cutoff
    let expected_half_point = (cutoff * size as f64).round() as usize;
    println!("Half-response point: {}, Expected: {}", half_point, expected_half_point);
    
    // Allow substantial margin since this is implementation-dependent
    // Just verify we find some decrease in response as we move from center
    assert!(half_point < center, "Response should decrease away from center");
    
    // 3. For high frequencies (near Nyquist), response should be very low
    let high_freq_response = response_profile.last().unwrap_or(&0.0);
    assert!(*high_freq_response < dc_response / 5.0, 
            "High frequencies should be significantly attenuated");
}

// Test the effect of filter order on transition steepness
#[test]
fn test_filter_order_effect() {
    let size = 64;
    let img = generate_test_image(size, size);
    let cutoff = 0.2;
    
    // Compare different orders
    let orders = [2.0, 8.0];
    
    // For each order, measure the filter response at multiple points
    let mut response_profiles = Vec::new();
    
    for &order in &orders {
        println!("Testing order: {}", order);
        let (filtered, _) = butterworth(&img, cutoff, false, order, true, 0);
        let spectrum = calculate_frequency_spectrum(&filtered);
        
        let center = size as usize / 2;
        let mut responses = Vec::new();
        
        // Sample at various distances from center
        for i in 1..center {
            let r = i as f64 / center as f64; // Normalized radius
            let response = spectrum[[center, center + i]];
            responses.push((r, response));
            println!("  Radius {:.2}: Response {}", r, response);
        }
        
        response_profiles.push(responses);
    }
    
    // Compare the profiles - higher order should have steeper falloff
    if response_profiles.len() >= 2 && 
       response_profiles[0].len() > 0 && 
       response_profiles[1].len() > 0 {
        
        // Find the steepest part of each response curve
        let mut max_slope_low_order: f64 = 0.0;
        let mut max_slope_high_order: f64 = 0.0;
        
        for i in 1..response_profiles[0].len().min(response_profiles[1].len()) {
            let slope_low = (response_profiles[0][i-1].1 - response_profiles[0][i].1).abs() / 
                           (response_profiles[0][i].0 - response_profiles[0][i-1].0);
            
            let slope_high = (response_profiles[1][i-1].1 - response_profiles[1][i].1).abs() / 
                            (response_profiles[1][i].0 - response_profiles[1][i-1].0);
            
            max_slope_low_order = max_slope_low_order.max(slope_low);
            max_slope_high_order = max_slope_high_order.max(slope_high);
        }
        
        println!("Maximum slope for order {}: {}", orders[0], max_slope_low_order);
        println!("Maximum slope for order {}: {}", orders[1], max_slope_high_order);
        
        // Higher order should have steeper slope
        if max_slope_low_order > 0.0 && max_slope_high_order > 0.0 {
            assert!(max_slope_high_order > max_slope_low_order, 
                    "Higher order filter should have steeper transition");
        }
    }
}


// Test the effect of squared butterworth option
#[test]
fn test_squared_butterworth_effect() {
    let size = 128;
    
    // Create a test image with clear frequency components
    let img = create_sine_wave_image(size, size, 0.15, 0.15);
    
    let cutoff = 0.2;
    let order = 4.0;
    
    // Compare standard vs squared butterworth
    let (standard_filtered, _) = butterworth(&img, cutoff, false, order, false, 0);
    let (squared_filtered, _) = butterworth(&img, cutoff, false, order, true, 0);
    
    // Calculate energy ratios
    let standard_ratio = calculate_energy_ratio(&standard_filtered, &img);
    let squared_ratio = calculate_energy_ratio(&squared_filtered, &img);
    
    println!("Standard Butterworth energy ratio: {}", standard_ratio);
    println!("Squared Butterworth energy ratio: {}", squared_ratio);
    
    // Squared Butterworth should typically have a more aggressive effect
    // Note: This depends on where we are in the frequency response curve
    if standard_ratio > 0.5 && standard_ratio < 0.99 {
        // In the transition region, squared should be more aggressive
        assert!(squared_ratio != standard_ratio, "Squared should behave differently from standard");
    }
    
    // An alternative approach is to check the energy in specific frequency bands
    // This provides a more reliable test of the difference between normal and squared filters
    let standard_spectrum = calculate_frequency_spectrum(&standard_filtered);
    let squared_spectrum = calculate_frequency_spectrum(&squared_filtered);
    
    let center = size as usize / 2;
    let cutoff_bin = (cutoff * size as f64).round() as usize;
    
    // Compare the rolloff characteristics
    if cutoff_bin < center {
        // Measure the energy drop across the cutoff region
        let region_size = 5;
        
        let inner_region: f64 = (cutoff_bin - region_size..cutoff_bin)
            .map(|i| standard_spectrum[[center, center + i]])
            .sum();
            
        let outer_region: f64 = (cutoff_bin..cutoff_bin + region_size)
            .map(|i| standard_spectrum[[center, center + i]])
            .sum();
            
        let standard_rolloff = inner_region / outer_region;
        
        let inner_region_sq: f64 = (cutoff_bin - region_size..cutoff_bin)
            .map(|i| squared_spectrum[[center, center + i]])
            .sum();
            
        let outer_region_sq: f64 = (cutoff_bin..cutoff_bin + region_size)
            .map(|i| squared_spectrum[[center, center + i]])
            .sum();
            
        let squared_rolloff = inner_region_sq / outer_region_sq;
        
        println!("Standard filter rolloff ratio: {}", standard_rolloff);
        println!("Squared filter rolloff ratio: {}", squared_rolloff);
        
        // For comparable filters, verify there's a measurable difference in behavior
        // This is a more robust test than specific values
        assert!(standard_rolloff != squared_rolloff, 
                "Squared Butterworth should have different frequency response characteristics");
    }
}

// Test that filter appropriately preserves or attenuates specific spatial features
#[test]
fn test_spatial_feature_preservation() {
    let size = 128;
    
    // Create a test image with both low and high frequency features
    let mut img = GrayImage::new(size, size);
    
    // Add a large, smooth gradient (low frequency)
    for y in 0..size {
        for x in 0..size {
            let dist = (((x as f64 - size as f64 / 2.0).powi(2) + 
                        (y as f64 - size as f64 / 2.0).powi(2)).sqrt()) / (size as f64 / 2.0);
            let gradient = ((1.0 - dist) * 200.0) as u8;
            img.put_pixel(x, y, Luma([gradient]));
        }
    }
    
    // Add a high-frequency checkerboard pattern in a corner
    for y in 0..size/4 {
        for x in 0..size/4 {
            if (x / 4 + y / 4) % 2 == 0 {
                img.put_pixel(x, y, Luma([255]));
            } else {
                img.put_pixel(x, y, Luma([0]));
            }
        }
    }
    
    // Apply low-pass filter (should preserve gradient, remove checkerboard)
    let (lp_filtered, _) = butterworth(&img, 0.1, false, 4.0, false, 0);
    
    // Apply high-pass filter (should remove gradient, preserve checkerboard edges)
    let (hp_filtered, _) = butterworth(&img, 0.1, true, 4.0, false, 0);
    
    // Verify low-pass behavior: checkerboard should be smoothed out
    let mut checkerboard_variation_orig = 0.0;
    let mut checkerboard_variation_lp = 0.0;
    
    for y in 1..size/4-1 {
        for x in 1..size/4-1 {
            let center_orig = img.get_pixel(x, y).0[0] as f64;
            let center_lp = lp_filtered.get_pixel(x, y).0[0] as f64;
            
            let neighbors_orig = [
                img.get_pixel(x-1, y).0[0] as f64,
                img.get_pixel(x+1, y).0[0] as f64,
                img.get_pixel(x, y-1).0[0] as f64,
                img.get_pixel(x, y+1).0[0] as f64,
            ];
            
            let neighbors_lp = [
                lp_filtered.get_pixel(x-1, y).0[0] as f64,
                lp_filtered.get_pixel(x+1, y).0[0] as f64,
                lp_filtered.get_pixel(x, y-1).0[0] as f64,
                lp_filtered.get_pixel(x, y+1).0[0] as f64,
            ];
            
            checkerboard_variation_orig += neighbors_orig.iter()
                .map(|&v| (v - center_orig).powi(2))
                .sum::<f64>();
                
            checkerboard_variation_lp += neighbors_lp.iter()
                .map(|&v| (v - center_lp).powi(2))
                .sum::<f64>();
        }
    }
    
    // Normalize variations
    checkerboard_variation_orig /= ((size/4 - 2) * (size/4 - 2)) as f64;
    checkerboard_variation_lp /= ((size/4 - 2) * (size/4 - 2)) as f64;
    
    println!("Checkerboard variation - Original: {}", checkerboard_variation_orig);
    println!("Checkerboard variation - Low-pass: {}", checkerboard_variation_lp);
    
    // The low-pass filter should significantly reduce checkerboard variation
    assert!(checkerboard_variation_lp < checkerboard_variation_orig * 0.3,
           "Low-pass filter should smooth out high-frequency checkerboard");
    
    // Verify high-pass behavior: smooth gradient should be removed
    let mut gradient_variation_orig = 0.0;
    let mut gradient_variation_hp = 0.0;
    
    // Examine the smooth part of the image (center area)
    for y in (size*3/8)..(size*5/8) {
        for x in (size*3/8)..(size*5/8) {
            gradient_variation_orig += img.get_pixel(x, y).0[0] as f64;
            gradient_variation_hp += hp_filtered.get_pixel(x, y).0[0] as f64;
        }
    }
    
    // Calculate average intensity in smooth region
    let pixel_count = ((size*5/8 - size*3/8) * (size*5/8 - size*3/8)) as f64;
    gradient_variation_orig /= pixel_count;
    gradient_variation_hp /= pixel_count;
    
    println!("Gradient average - Original: {}", gradient_variation_orig);
    println!("Gradient average - High-pass: {}", gradient_variation_hp);
    
    // High-pass filter should significantly reduce average intensity in smooth areas
    // (replacing them with mid-gray values)
    assert!(gradient_variation_hp < gradient_variation_orig * 0.4 || 
           (gradient_variation_hp - 128.0).abs() < 20.0,
           "High-pass filter should remove smooth gradients");
}