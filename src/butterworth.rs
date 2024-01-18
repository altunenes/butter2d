// butterworth.rs
use ndarray::{Array, ArrayD, IxDyn};
use std::f64::consts::PI;
use rustfft::{FftPlanner, num_complex::Complex};
use image::{GrayImage, ImageBuffer, Luma};
use ndarray::{Array2, s};
use fft2d::nalgebra::{fft_2d, ifft_2d};
use image::imageops::resize;
use nalgebra::DMatrix;
use nalgebra::Matrix;

/// Create a N-dimensional Butterworth mask for an FFT
/// 
/// # Arguments
/// * `shape` - Shape of the n-dimensional FFT and mask.
/// * `factor` - Fraction of mask dimensions where the cutoff should be.
/// * `order` - Controls the slope in the cutoff region.
/// * `high_pass` - Whether the filter is high pass or low pass.
/// * `real` - Whether the FFT is of a real or complex image.
/// * `squared_butterworth` - If true, the square of the Butterworth filter is used.
/// 
/// # Returns
/// * `ArrayD<Complex<f64>>` - The FFT mask.
pub fn get_nd_butterworth_filter(
    shape: &[usize],
    factor: f64,
    order: f64,
    high_pass: bool,
    real: bool,
    squared_butterworth: bool,
) -> ArrayD<Complex<f64>> {
    // Create ranges for each axis
    let ranges: Vec<ArrayD<f64>> = shape
        .iter()
        .enumerate()
        .map(|(i, &d)| {
            let axis_range: ArrayD<f64> = Array::linspace(
                -(d as f64 - 1.0) / 2.0,
                (d as f64 - 1.0) / 2.0,
                d
            ).into_shape(IxDyn(&[d])).unwrap() / (d as f64 * factor);
            
            if i == shape.len() - 1 && real {
                // Convert the sliced array back to ArrayD with dynamic dimensions
                axis_range.slice(s![..d / 2 + 1]).to_owned().into_dyn().mapv(|x| x.powi(2))
            } else {
                axis_range.mapv(|x| x.powi(2))
            }
            
        })
        .collect();
    // Calculate squared Euclidean distance grid (q2)
    let q2 = ranges
    .iter()
    .fold(Array::zeros(IxDyn(&[shape.len()])), |acc, arr| acc + arr);
    // Calculate the Butterworth filter
    let ones: ArrayD<f64> = Array::ones(q2.dim());
    let mut wfilt = Array::from_elem(q2.dim(), Complex::new(1.0, 0.0)) 
        / (ones + q2.mapv(|x: f64| x.powf(order))); // Use the f64 ones array
    
    if high_pass {
        wfilt.zip_mut_with(&q2.mapv(|x| Complex::new(x, 0.0)), |a, &b| *a *= b);
    }
    if !squared_butterworth {
        wfilt.mapv_inplace(|x| x.sqrt());
    }
    
    wfilt
}

/// Pad the image with edge value extension.
fn pad_image(image: &GrayImage, npad: usize) -> DMatrix<Complex<f64>> {
    let (width, height) = image.dimensions();
    let padded_width = width + 2 * npad as u32;
    let padded_height = height + 2 * npad as u32;
    let padded_image = resize(image, padded_width, padded_height, image::imageops::FilterType::Nearest);

    DMatrix::from_iterator(
        padded_height as usize,
        padded_width as usize,
        padded_image
            .pixels()
            .map(|p| Complex::new(p.0[0] as f64, 0.0))
    )
}

/// Apply FFT, Butterworth filter, and inverse FFT.
fn apply_fft_and_filter(
    padded_image: &DMatrix<Complex<f64>>,
    butterworth_filter: &DMatrix<Complex<f64>>, // Change the type to DMatrix
) -> GrayImage {
    let fft_image = fft_2d(padded_image.clone());
    
    // Ensure the dimensions match
    if fft_image.nrows() != butterworth_filter.nrows() || fft_image.ncols() != butterworth_filter.ncols() {
        panic!("Mismatch in dimensions between FFT image and Butterworth filter");
    }

    // Apply the filter directly without converting to another type
    let filtered_image = fft_image.zip_map(butterworth_filter, |img_val, filter_val| img_val * filter_val);
    let ifft_image = ifft_2d(filtered_image);

    // Normalize and convert to grayscale
    let max_val = ifft_image.iter().map(|c| c.norm()).fold(0.0, f64::max);
    GrayImage::from_raw(
        ifft_image.ncols().try_into().unwrap(),
        ifft_image.nrows().try_into().unwrap(),
        ifft_image.iter().map(|c| (c.norm() / max_val * 255.0) as u8).collect()
    ).unwrap()
}
/// Apply a Butterworth filter to enhance high or low frequency features.
pub fn butterworth(
    image: &GrayImage,
    cutoff_frequency_ratio: f64,
    high_pass: bool,
    order: f64,
    squared_butterworth: bool,
    npad: usize,
) -> GrayImage {
    if cutoff_frequency_ratio < 0.0 || cutoff_frequency_ratio > 0.5 {
        panic!("cutoff_frequency_ratio should be in the range [0, 0.5]");
    }

    // Pad the image
    let padded_image = pad_image(image, npad);

    // Calculate the shape for FFT
    let fft_shape = &[padded_image.nrows(), padded_image.ncols()];
    // Generate Butterworth filter using ndarray
    let butterworth_filter_ndarray = get_nd_butterworth_filter(
        fft_shape,
        cutoff_frequency_ratio,
        order,
        high_pass,
        true, // Assuming the image is real
        squared_butterworth,
    );

    // Convert ndarray to DMatrix
    let butterworth_filter = DMatrix::from_iterator(
        fft_shape[0],
        fft_shape[1],
        butterworth_filter_ndarray.iter().cloned().map(|x| Complex::new(x.re, x.im))
    );

    // Apply FFT, filter, and inverse FFT
    apply_fft_and_filter(&padded_image, &butterworth_filter)
}