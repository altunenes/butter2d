use ndarray::{Array, ArrayD, IxDyn,Zip};
use std::f64::consts::PI;
use rustfft::{FftPlanner, num_complex::Complex};
use image::{GrayImage, ImageBuffer, Luma};
use ndarray::{Array2, s};
use fft2d::nalgebra::{fft_2d, ifft_2d};
use image::imageops::resize;
use nalgebra::DMatrix;
use nalgebra::Matrix;
use itertools::Itertools;

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
    let mut ranges = Vec::new();

    for &d in shape.iter() {
        let axis_range: Vec<f64> = Array::linspace(-(d as f64 - 1.0) / 2.0, (d as f64 - 1.0) / 2.0, d)
            .into_raw_vec()
            .iter()
            .map(|&x| x.powi(2))
            .collect();
        ranges.push(axis_range);
    }

    // Generate the Cartesian product of ranges
    let grid: Vec<Vec<f64>> = ranges[0].iter()
        .cartesian_product(ranges[1].iter())
        .map(|(&x, &y)| vec![x, y])
        .collect();

    // Convert grid to ndarray
    let grid_shape = shape.iter().map(|&d| d).collect::<Vec<_>>();
    let q2 = Array::from_shape_vec(IxDyn(&grid_shape), grid.iter().map(|coords| coords.iter().sum()).collect::<Vec<_>>()).unwrap();

    // Calculate the Butterworth filter
    let ones: ArrayD<f64> = Array::ones(q2.dim());
    let mut wfilt = Array::from_elem(q2.dim(), Complex::new(1.0, 0.0)) / (ones.mapv(|x| Complex::new(x, 0.0)) + q2.mapv(|x: f64| x.powf(order)));
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
    println!("Original image dimensions: ({}, {})", width, height);

    // Calculate the nearest power of 2 for each dimension
    let padded_width = (width as usize + 2 * npad).next_power_of_two() as u32;
    let padded_height = (height as usize + 2 * npad).next_power_of_two() as u32;
    println!("Padded image dimensions: ({}, {})", padded_width, padded_height);

    let padded_image = resize(image, padded_width, padded_height, image::imageops::FilterType::Nearest);

    DMatrix::from_iterator(
        padded_height as usize,
        padded_width as usize,
        padded_image.pixels().map(|p| Complex::new(p.0[0] as f64, 0.0))
    )
}

/// Apply FFT, Butterworth filter, and inverse FFT.
fn apply_fft_and_filter(
    padded_image: &DMatrix<Complex<f64>>,
    butterworth_filter: &DMatrix<Complex<f64>>,
) -> GrayImage {
    let fft_image = fft_2d(padded_image.clone());

    // Debugging: Print dimensions of FFT image and Butterworth filter
    println!("FFT image dimensions: ({}, {})", fft_image.nrows(), fft_image.ncols());
    println!("Butterworth filter dimensions: ({}, {})", butterworth_filter.nrows(), butterworth_filter.ncols());

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
     // Pad the image
     let padded_image = pad_image(image, npad);

     // Calculate the shape for FFT
     let fft_shape = &[padded_image.nrows(), padded_image.ncols()];
     println!("FFT shape for Butterworth filter: {:?}", fft_shape);
 
     // Generate Butterworth filter using ndarray
     let butterworth_filter_ndarray = get_nd_butterworth_filter(
         fft_shape,
         cutoff_frequency_ratio,
         order,
         high_pass,
         true, // Assuming the image is real
         squared_butterworth,
     );

     println!("ndarray Butterworth filter dimensions: {:?}", butterworth_filter_ndarray.dim());

     // Convert ndarray to DMatrix
     let butterworth_filter = DMatrix::from_iterator(
         fft_shape[0],
         fft_shape[1],
         butterworth_filter_ndarray.iter().cloned().map(|x| Complex::new(x.re, x.im))
     );
 
     // Debug: Print dimensions of the DMatrix filter
     println!("DMatrix Butterworth filter dimensions: ({}, {})", butterworth_filter.nrows(), butterworth_filter.ncols());
 
     // Apply FFT, filter, and inverse FFT
     apply_fft_and_filter(&padded_image, &butterworth_filter)
 }