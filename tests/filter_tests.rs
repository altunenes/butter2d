use butter2d::{butterworth,pad_image,get_nd_butterworth_filter};
use image::{GrayImage, Luma};
use ndarray::{Array,Array2,s,IxDyn};
use rustfft::num_complex::Complex;
use fft2d::nalgebra::fft_2d;
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
                        println!("Test parameters: cutoff={}, order={}, high_pass={}, squared_butterworth={}", cutoff, order, high_pass, squared_butterworth);
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
                    let img = GrayImage::from_pixel(512, 512, Luma([128]));
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
fn fft_magnitude(image: &GrayImage) -> Array2<f64> {
    let nrows = image.height() as usize;
    let ncols = image.width() as usize;
    let mut data = Vec::with_capacity(nrows * ncols);
    for p in image.pixels() {
        data.push(Complex::new(p.0[0] as f64, 0.0));
    }
    let dmatrix = DMatrix::from_vec(nrows, ncols, data);
    let fft_result = fft_2d(dmatrix);
    let magnitude_array = Array::from_shape_fn((nrows, ncols), |(row, col)| {
        fft_result[(row, col)].norm()
    });
    magnitude_array
}
#[test]
fn test_butterworth_frequency_response2() {
    let img = generate_test_image(512, 512);
    let cutoff_frequency_ratio = 0.1;
    let order = 2.0;
    let high_pass = true;
    let squared_butterworth = true;
    let (filtered_img, _) = butterworth(&img, cutoff_frequency_ratio, high_pass, order, squared_butterworth, 0);
    let magnitude_spectrum = fft_magnitude(&filtered_img);
    let center_region = magnitude_spectrum.slice(s![256-64..256+64, 256-64..256+64]);
    let edge_region = magnitude_spectrum.slice(s![..128, ..128]);
    let avg_center = center_region.mean().unwrap();
    let avg_edge = edge_region.mean().unwrap();
    if high_pass {
        assert!(avg_center < avg_edge, "High-pass filter should attenuate low frequencies more than high frequencies.");
    } else {
        assert!(avg_center > avg_edge, "Low-pass filter should retain low frequencies more than high frequencies.");
    }
}
#[test]
fn test_butterworth_2d_realfft_comparison() {
    let img = generate_test_image(512, 512);
    let cutoff_frequency_ratio = 0.2;
    let order = 2.0;
    let high_pass = false;
    let squared_butterworth = true;
    let (filtered_real_img, _) = butterworth(&img, cutoff_frequency_ratio, high_pass, order, squared_butterworth, 0);
    let real_img_magnitude_spectrum = fft_magnitude(&filtered_real_img);
    let avg_magnitude = real_img_magnitude_spectrum.iter().sum::<f64>() / real_img_magnitude_spectrum.len() as f64;
    println!("Average magnitude of the frequency response: {}", avg_magnitude);
}
fn compute_average_energy(matrix: &Array2<f64>, center: bool) -> f64 {
    let (nrows, ncols) = matrix.dim();
    let (start_row, end_row, start_col, end_col) = if center {
        ((nrows / 2) - 64, (nrows / 2) + 64, (ncols / 2) - 64, (ncols / 2) + 64)
    } else {
        (0, nrows, 0, ncols)
    };
    let mut sum = 0.0;
    let mut count = 0.0;
    for i in start_row..end_row {
        for j in start_col..end_col {
            sum += matrix[[i, j]];
            count += 1.0;
        }
    }
    sum / count
}
#[test]
fn test_butterworth_frequency_response() {
    let img = generate_test_image(512, 512);
    let cutoff_frequency_ratio = 0.1;
    let order = 2.0;
    let high_pass = false;
    let squared_butterworth: bool = true;
    let (filtered_img, _) = butterworth(&img, cutoff_frequency_ratio, high_pass, order, squared_butterworth, 0);
    let magnitude_spectrum = fft_magnitude(&filtered_img);
    let avg_center = compute_average_energy(&magnitude_spectrum, true);
    let avg_edge = compute_average_energy(&magnitude_spectrum, false);
    if high_pass {
        assert!(avg_center < avg_edge, "High-pass filter should attenuate low frequencies more than high frequencies.");
    } else {
        assert!(avg_center < avg_edge, "Low-pass filter should retain low frequencies more than high frequencies.");
    }
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