use ndarray::{Array, ArrayD, IxDyn};
use rustfft::num_complex::Complex;
use image::GrayImage;
use rustfft::FftPlanner;
use nalgebra::DMatrix;

/// Performs 2D Fast Fourier Transform on a matrix
pub fn fft2d(matrix: &DMatrix<Complex<f64>>) -> DMatrix<Complex<f64>> {
    let rows = matrix.nrows();
    let cols = matrix.ncols();
    let mut planner = FftPlanner::new();
    
    // Create a new matrix to store the result
    let mut result = matrix.clone();
    
    // Perform FFT on each row
    for i in 0..rows {
        let mut row_data: Vec<Complex<f64>> = (0..cols).map(|j| matrix[(i, j)]).collect();
        let fft = planner.plan_fft_forward(cols);
        fft.process(&mut row_data);
        for j in 0..cols {
            result[(i, j)] = row_data[j];
        }
    }
    
    // Perform FFT on each column
    for j in 0..cols {
        let mut col_data: Vec<Complex<f64>> = (0..rows).map(|i| result[(i, j)]).collect();
        let fft = planner.plan_fft_forward(rows);
        fft.process(&mut col_data);
        for i in 0..rows {
            result[(i, j)] = col_data[i];
        }
    }
    
    result
}

/// Performs 2D Inverse Fast Fourier Transform on a matrix
pub fn ifft2d(matrix: &DMatrix<Complex<f64>>) -> DMatrix<Complex<f64>> {
    let rows = matrix.nrows();
    let cols = matrix.ncols();
    let mut planner = FftPlanner::new();
    
    // Create a new matrix to store the result
    let mut result = matrix.clone();
    
    // Perform IFFT on each row
    for i in 0..rows {
        let mut row_data: Vec<Complex<f64>> = (0..cols).map(|j| matrix[(i, j)]).collect();
        let ifft = planner.plan_fft_inverse(cols);
        ifft.process(&mut row_data);
        for j in 0..cols {
            result[(i, j)] = row_data[j] / Complex::new(cols as f64, 0.0);
        }
    }
    
    // Perform IFFT on each column
    for j in 0..cols {
        let mut col_data: Vec<Complex<f64>> = (0..rows).map(|i| result[(i, j)]).collect();
        let ifft = planner.plan_fft_inverse(rows);
        ifft.process(&mut col_data);
        for i in 0..rows {
            result[(i, j)] = col_data[i] / Complex::new(rows as f64, 0.0);
        }
    }
    
    result
}

/// Constructs an N-dimensional Butterworth filter mask for Fourier Transform operations.
///
/// This function generates a mask to be applied in the frequency domain, useful for
/// filtering images or signals to either enhance or suppress frequencies. The Butterworth
/// filter is known for its smooth frequency response and is parameterized to control
/// the sharpness of its cutoff.
///
/// # Arguments
/// * `shape` - A slice representing the dimensions of the n-dimensional FFT and the resulting mask.
///   Specifies the size of the output mask, matching the FFT's output dimensions.
/// * `factor` - A floating-point value representing the fraction of the mask dimensions at which
///   the cutoff frequency is set. It determines the boundary between the passband and the stopband
///   in normalized frequency units (0 to 1).
/// * `order` - A floating-point value that controls the steepness or slope of the filter's
///   transition from passband to stopband. Higher values result in a steeper transition.
/// * `high_pass` - A boolean indicating the filter type. If `true`, the filter is a high-pass filter
///   (attenuating frequencies below the cutoff). If `false`, it's a low-pass filter (attenuating
///   frequencies above the cutoff).
/// * `real` - A boolean indicating whether the FFT represents a real (`true`) or complex (`false`)
///   image or signal. This parameter may affect the symmetry and size of the output mask.
/// * `squared_butterworth` - A boolean specifying whether to use the square of the Butterworth
///   filter (`true`) or not (`false`). Using the square can alter the filter's characteristics,
///   particularly its sharpness and transition slope.
///
/// # Returns
/// * `ArrayD<Complex<f64>>` - An n-dimensional array containing the complex-valued Butterworth filter
///   mask. The mask is designed to be multiplied directly with the FFT of an image or signal.
/// # Notes
/// The Butterworth filter is widely used in signal processing and image processing due to its
/// maximally flat frequency response in the passband and no ripples in the stopband. The `order`
/// parameter can significantly affect the filter's performance, especially in terms of transition
/// bandwidth and attenuation rate. Careful selection of `factor` and `order` is recommended to
/// meet specific filtering requirements.
pub fn get_nd_butterworth_filter(
    shape: &[usize],
    factor: f64,
    order: f64,
    high_pass: bool,
    _real: bool,
    squared_butterworth: bool,
) -> ArrayD<Complex<f64>> {
    // Create a frequency domain grid
    let wfilt = Array::from_shape_fn(IxDyn(shape), |idx| {
        let len = shape.len();
        let mut distance2 = 0.0;
        for i in 0..len {
            let mut freq = idx[i] as f64;
            if freq > shape[i] as f64 / 2.0 {
                freq -= shape[i] as f64;
            }
            freq /= shape[i] as f64;
            distance2 += freq.powi(2);
        }
        let radius = distance2.sqrt();
        let response = 1.0 / (1.0 + (radius / factor).powf(order * 2.0));
        let response = if high_pass { 
            (radius / factor).powf(order * 2.0) / (1.0 + (radius / factor).powf(order * 2.0))
        } else { 
            response 
        };
        let response = if squared_butterworth { response.powi(2) } else { response };
        Complex::new(response, 0.0)
    });
    wfilt
}

/// Pads and resizes an image to the nearest powers of two dimensions.
///
/// This function is designed to prepare an image for Fourier Transform operations by padding its
/// dimensions to the nearest powers of two. Padding is symmetrically applied to both the width and
/// height of the image to minimize edge effects during FFT processing. The image is resized using
/// nearest neighbor interpolation to maintain the integrity of the original pixel values as closely
/// as possible.
///
/// # Arguments
/// * `image` - A reference to a `GrayImage` representing the input image to be padded and resized.
///   The `GrayImage` type is part of the `image` crate, which represents an image in grayscale format.
/// * `npad` - The number of pixels to symmetrically pad around the edges of the image before resizing.
///   This value is applied to all sides of the image, effectively increasing the total width and height
///   by `2 * npad`. The padding operation ensures that the final dimensions are suitable for FFT by
///   extending them to the nearest powers of two.
///
/// # Returns
/// * `DMatrix<Complex<f64>>` - A matrix of complex numbers where the real part represents the padded
///   and resized image, and the imaginary part is set to zero. This format is compatible with FFT
///   operations that require complex input.
pub fn pad_image(image: &GrayImage, npad: usize) -> (DMatrix<Complex<f64>>, (u32, u32)) {
    let (original_width, original_height) = image.dimensions();
    
    let padded_width = (original_width as usize + 2 * npad).next_power_of_two() as u32;
    let padded_height = (original_height as usize + 2 * npad).next_power_of_two() as u32;
    
    let mut padded_image = GrayImage::new(padded_width, padded_height);
    
    let x_offset = (padded_width - original_width) / 2;
    let y_offset = (padded_height - original_height) / 2;
    
    for y in 0..original_height {
        for x in 0..original_width {
            let pixel = image.get_pixel(x, y);
            padded_image.put_pixel(x + x_offset, y + y_offset, *pixel);
        }
    }
    let complex_matrix = DMatrix::from_iterator(
        padded_height as usize,
        padded_width as usize,
        padded_image.pixels().map(|p| Complex::new(p.0[0] as f64 / 255.0, 0.0))
    );
    (complex_matrix, (original_width, original_height))
}

// Function to extract the original portion from the filtered image
fn extract_original_portion(filtered_image: &GrayImage, original_dimensions: (u32, u32)) -> GrayImage {
    let (original_width, original_height) = original_dimensions;
    let (current_width, current_height) = filtered_image.dimensions();
    
    // If dimensions are already the same, return a clone
    if original_width == current_width && original_height == current_height {
        return filtered_image.clone();
    }
    
    // Calculate the extraction box (centered)
    let x_offset = (current_width - original_width) / 2;
    let y_offset = (current_height - original_height) / 2;
    
    // Extract the portion matching the original dimensions
    image::imageops::crop_imm(filtered_image, x_offset, y_offset, original_width, original_height)
        .to_image()
}


/// This function orchestrates the core steps in frequency domain filtering: it first applies FFT to
/// the input image, then applies the Butterworth filter mask, and finally performs an inverse FFT to
/// return the filtered image back to the spatial domain. A crucial normalization step is included after
/// the inverse FFT to ensure the output image has appropriate brightness levels.
///
/// # Arguments
/// * `padded_image` - A reference to a `DMatrix<Complex<f64>>` representing the padded input image
///   in the complex domain, ready for FFT processing. The matrix should have dimensions that are powers
///   of two for optimal FFT performance.
/// * `butterworth_filter` - A reference to a `DMatrix<Complex<f64>>` representing the Butterworth filter
///   mask. This matrix should have the same dimensions as `padded_image` to ensure element-wise multiplication
///   is properly aligned.
///
/// # Returns
/// * `GrayImage` - The filtered image, transformed back into the spatial domain and normalized to
///   utilize the full range of grayscale values (0-255).
fn apply_fft_and_filter(
    padded_image: &DMatrix<Complex<f64>>,
    butterworth_filter: &DMatrix<Complex<f64>>,
) -> GrayImage {
    // Apply FFT to the image
    let fft_image = fft2d(padded_image);
    
    // Check dimensions match
    assert_eq!(fft_image.nrows(), butterworth_filter.nrows());
    assert_eq!(fft_image.ncols(), butterworth_filter.ncols());
    
    // Apply the filter in frequency domain
    let mut filtered_image = DMatrix::zeros(fft_image.nrows(), fft_image.ncols());
    for i in 0..fft_image.nrows() {
        for j in 0..fft_image.ncols() {
            filtered_image[(i, j)] = fft_image[(i, j)] * butterworth_filter[(i, j)];
        }
    }
    
    // Apply inverse FFT to get back to spatial domain
    let ifft_image = ifft2d(&filtered_image);
    
    // Extract real parts for image creation
    let real_ifft: Vec<f64> = ifft_image.iter().map(|c| c.re).collect();
    
    // Find min and max values for normalization
    let min_val = real_ifft.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_val = real_ifft.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    
    // Check if normalization is necessary
    if (max_val - min_val) > f64::EPSILON {
        // Apply normalization only if there's significant variation in the output
        let scale = 255.0 / (max_val - min_val);
        GrayImage::from_raw(
            ifft_image.ncols() as u32,
            ifft_image.nrows() as u32,
            real_ifft.iter().map(|&val| {
                let scaled_val = ((val - min_val) * scale).min(255.0).max(0.0) as u8;
                scaled_val
            }).collect()
        ).unwrap()
    } else {
        // If the output is essentially uniform, treat it as zero
        GrayImage::new(ifft_image.ncols() as u32, ifft_image.nrows() as u32)
    } 
}

/// Applies a Butterworth filter to an image to enhance or suppress frequency components.
///
/// This function performs spatial domain to frequency domain transformation, applies a Butterworth
/// filter to enhance high or low frequency features based on the specified parameters, and then
/// transforms the result back to the spatial domain. The function is designed to work with grayscale
/// images and utilizes FFT (Fast Fourier Transform) for efficient frequency domain processing.
///
/// # Arguments
/// * `image` - A reference to a `GrayImage` representing the input image. The `GrayImage` is part of
///   the `image` crate and should contain grayscale pixel values.
/// * `cutoff_frequency_ratio` - A floating-point value between 0.0 and 0.5 that determines the cutoff
///   frequency as a ratio of the Nyquist frequency, which is half of the sampling rate according to the
///   Nyquist criterion. This parameter thus defines the boundary between the filter's passband and stopband
///   in terms of cycles per pixel in the spatial frequency domain. For a low-pass filter, selecting a cutoff
///   frequency ratio of 0.5 results in a non-aggressive filter, affecting only the highest frequencies just
///   below the Nyquist limit. The aggressiveness of the low-pass filter increases as the cutoff frequency ratio
///   approaches 0.0, progressively attenuating a broader range of higher frequencies. Conversely, for a high-pass
///   filter, a cutoff frequency ratio of 0.5 corresponds to a very aggressive filter, significantly attenuating
///   frequencies across most of the spectrum, while a ratio near 0.0 results in a less aggressive filter,
///   primarily affecting only the lowest frequencies and preserving most of the higher frequencies.
/// * `high_pass` - A boolean indicating the type of filter to apply. If `true`, the function applies a
///   high-pass filter, attenuating frequencies below the cutoff. If `false`, a low-pass filter is applied,
///   attenuating frequencies above the cutoff.
/// * `order` - A floating-point value that specifies the order of the Butterworth filter. The order
///   determines the steepness of the filter's transition between the passband and stopband.
/// * `squared_butterworth` - A boolean indicating whether to square the Butterworth filter response.
///   Squaring the filter can sharpen the transition between the passband and stopband.
/// * `npad` - The number of pixels by which to pad the input image before applying the FFT. Padding can
///   help reduce edge effects and improve the filter's performance.
///
/// # Returns
/// A tuple containing two elements:
/// * `GrayImage` - The filtered image, returned to the spatial domain and normalized to utilize the full
///   grayscale range. The normalization ensures the image's visibility and contrast are enhanced according
///   to the filter's effect.
/// * `DMatrix<Complex<f64>>` - The Butterworth filter used in the frequency domain. This matrix represents
///   the filter mask applied to the FFT of the input image. It can be useful for analysis or applying the
///   same filter to multiple images.
///
/// # Panics
/// The function panics if the `cutoff_frequency_ratio` is not in the range [0.0, 0.5], ensuring that
/// filter parameters are within acceptable bounds for meaningful frequency domain processing.
pub fn butterworth(
    image: &GrayImage,
    cutoff_frequency_ratio: f64,
    high_pass: bool,
    order: f64,
    squared_butterworth: bool,
    npad: usize,
) -> (GrayImage, DMatrix<Complex<f64>>) {
    if cutoff_frequency_ratio < 0.0 || cutoff_frequency_ratio > 0.5 {
        panic!("cutoff_frequency_ratio should be in the range [0, 0.5]");
    }
    
    // Pad the image and prepare for FFT
    let (padded_image, original_dimensions) = pad_image(image, npad);
    let fft_shape = &[padded_image.nrows(), padded_image.ncols()];
    
    // Generate the Butterworth filter in frequency domain
    // The filter needs to match the padded dimensions, not the original
    let butterworth_filter_ndarray = get_nd_butterworth_filter(
        fft_shape,
        cutoff_frequency_ratio,
        order,
        high_pass,
        true,
        squared_butterworth,
    );
    
    // Convert ndarray filter to DMatrix for use with our FFT implementation
    let butterworth_filter = DMatrix::from_iterator(
        fft_shape[0],
        fft_shape[1],
        butterworth_filter_ndarray.iter().cloned()
    );
    
    // Apply FFT, filter, and inverse FFT
    let filtered_image = apply_fft_and_filter(&padded_image, &butterworth_filter);
    // Extract the portion corresponding to the original image dimensions
    let original_portion = extract_original_portion(&filtered_image, original_dimensions);
    
    // Return both the image and the filter used
    (original_portion, butterworth_filter)
}
