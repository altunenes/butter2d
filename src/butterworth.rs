use ndarray::{Array, Array1,Axis,ArrayD, IxDyn,Zip};
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

    let mut grid = Array::zeros(IxDyn(&[shape[0], shape[1]]));
    let (cx, cy) = (shape[0] / 2, shape[1] / 2);
    for (x, y) in itertools::iproduct!(0..shape[0], 0..shape[1]) {
        let dx = (x as f64 - cx as f64).powi(2);
        let dy = (y as f64 - cy as f64).powi(2);
        grid[[x, y]] = Complex::new((dx + dy).powf(order / 2.0), 0.0);
    }

    // Convert grid to ndarray
    let grid_shape = shape.iter().map(|&d| d).collect::<Vec<_>>();
    // Correctly create a frequency grid that is centered
    let mut q2 = Array::zeros(IxDyn(&[shape[0], shape[1]]));
    let (cx, cy) = ((shape[0] / 2) as f64, (shape[1] / 2) as f64);
    for (x, y) in itertools::iproduct!(0..shape[0], 0..shape[1]) {
        let dx = ((x as f64 - cx) / (shape[0] as f64)).powi(2);
        let dy = ((y as f64 - cy) / (shape[1] as f64)).powi(2);
        let distance = dx + dy; // This is the squared distance from the center of the frequency domain
        q2[[x, y]] = Complex::new(distance, 0.0);
    }
    q2 = q2.mapv_into(|x| x.powf(order));

    // Calculate the Butterworth filter using the correct grid
    let ones = Array::from_elem(q2.dim(), Complex::new(1.0, 0.0));
    // Avoid the type mismatch by ensuring the types match for the addition
    let denominator = ones.clone() + (&q2 * factor.powi(2 * order as i32));
    let mut wfilt = Array::from_shape_fn(IxDyn(shape), |idx| {
        let len = shape.len();
        let mut distance2 = 0.0;
        for i in 0..len {
            // Normalize frequency range from -0.5 to 0.5
            let mut freq = idx[i] as f64;
            if freq > shape[i] as f64 / 2.0 {
                freq -= shape[i] as f64;
            }
            freq /= shape[i] as f64;
            distance2 += freq.powi(2);
        }
        // Calculate the radius in the frequency domain
        let radius = distance2.sqrt();
        // Calculate the Butterworth filter response for this frequency
        let response = 1.0 / (1.0 + (radius / factor).powf(order * 2.0));
        // Determine high-pass or low-pass response
        let response = if high_pass { 1.0 - response } else { response };
        // Optionally square the response
        let response = if squared_butterworth { response.powi(2) } else { response };
        Complex::new(response, 0.0)
    });
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
        padded_image.pixels().map(|p| Complex::new(p.0[0] as f64 / 255.0, 0.0))
    
    )
}

/// Apply FFT, Butterworth filter, and inverse FFT.
fn apply_fft_and_filter(
    padded_image: &DMatrix<Complex<f64>>,
    butterworth_filter: &DMatrix<Complex<f64>>,
) -> GrayImage {
    let fft_image = fft_2d(padded_image.clone());

    // Ensure dimensions match
    assert_eq!(fft_image.nrows(), butterworth_filter.nrows());
    assert_eq!(fft_image.ncols(), butterworth_filter.ncols());

    // Apply the filter
    let filtered_image = fft_image.zip_map(butterworth_filter, |img_val, filter_val| img_val * filter_val);

    // Perform inverse FFT
    let ifft_image = ifft_2d(filtered_image);

    // Normalize the real part of the inverse FFT to the range [0, 255]
    let real_ifft: Vec<f64> = ifft_image.iter().map(|c| c.re).collect();
    let min_val = *real_ifft.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let max_val = *real_ifft.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let scale = 255.0 / (max_val - min_val);

    GrayImage::from_raw(
        ifft_image.ncols() as u32,
        ifft_image.nrows() as u32,
        real_ifft.iter().map(|&val| {
            let scaled_val = (val - min_val) * scale;
            scaled_val.min(255.0).max(0.0) as u8 // Clamp the value to [0, 255]
        }).collect()
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
) -> (GrayImage, DMatrix<Complex<f64>>) { // Return type changed to a tuple
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
     let final_image = apply_fft_and_filter(&padded_image, &butterworth_filter);

     // Return both the final image and the filter used
     (final_image, butterworth_filter) // Return a tuple containing the image and the filter
  }

 // Assuming you have a function to visualize or save a matrix to an image
 pub fn visualize_filter(filter: &DMatrix<Complex<f64>>) {
    let max_magnitude = filter.iter().map(|&f| f.norm_sqr()).fold(0.0, f64::max).sqrt();
    let image_data: Vec<u8> = filter.iter().map(|&f| {
        let magnitude = f.norm();
        let normalized_magnitude = (magnitude / max_magnitude).sqrt(); // Use sqrt for better visualization contrast
        (normalized_magnitude * 255.0) as u8
    }).collect();

    // Create and save the image
    let filter_image = GrayImage::from_raw(filter.nrows() as u32, filter.ncols() as u32, image_data).unwrap();
    filter_image.save("butterworth_filter_visualization.png").expect("Failed to save filter visualization");
}