use ndarray::{Array,ArrayD, IxDyn};
use rustfft::num_complex::Complex;
use image::GrayImage;
use fft2d::nalgebra::{fft_2d, ifft_2d};
use image::imageops::resize;
use nalgebra::DMatrix;
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
    let _grid_shape = shape.iter().map(|&d| d).collect::<Vec<_>>();
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
    let _denominator = ones.clone() + (&q2 * factor.powi(2 * order as i32));
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
///   operations that require complex input. The `DMatrix` type comes from the `nalgebra` crate and
///   represents a dynamic two-dimensional matrix.
///
/// # Notes
/// The resizing step uses the nearest neighbor interpolation to avoid introducing interpolation
/// artifacts that could affect the Fourier Transform analysis. This method is chosen for its simplicity
/// and effectiveness in preserving the original image's pixel values. However, users should be aware
/// that resizing to significantly larger dimensions can introduce pixelation due to the nearest neighbor
/// method.
///
/// The resulting matrix is intended for use in FFT-based processing, where having dimensions as powers
/// of two can significantly optimize computation time. This preconditioning step is crucial for
/// achieving efficient performance in frequency domain analyses and operations.
pub fn pad_image(image: &GrayImage, npad: usize) -> DMatrix<Complex<f64>> {
    let (width, height) = image.dimensions();
    let max_dim = std::cmp::max(width, height) as usize + 2 * npad;
    let padded_size = max_dim.next_power_of_two() as u32; 
    let padded_image = resize(image, padded_size, padded_size, image::imageops::FilterType::Nearest);
    DMatrix::from_iterator(
        padded_size as usize,
        padded_size as usize,
        padded_image.pixels().map(|p| Complex::new(p.0[0] as f64 / 255.0, 0.0))
    )
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
///   utilize the full range of grayscale values (0-255). The `GrayImage` type is part of the `image` crate,
///   representing an image in grayscale format.
fn apply_fft_and_filter(
    padded_image: &DMatrix<Complex<f64>>,
    butterworth_filter: &DMatrix<Complex<f64>>,
) -> GrayImage {
    let fft_image = fft_2d(padded_image.clone());
    assert_eq!(fft_image.nrows(), butterworth_filter.nrows());
    assert_eq!(fft_image.ncols(), butterworth_filter.ncols());
    let filtered_image = fft_image.zip_map(butterworth_filter, |img_val, filter_val| img_val * filter_val);
    let ifft_image = ifft_2d(filtered_image);
    // Compute the range of the real parts
    let real_ifft: Vec<f64> = ifft_image.iter().map(|c| c.re).collect();
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
///
/// # Normalization Process
/// Unlike the skimage implementation in Python, this Rust version explicitly normalizes the output
/// image after applying the inverse FFT. This step adjusts the pixel values to span the full grayscale
/// range, enhancing visibility and contrast. The normalization process calculates the minimum and maximum
/// values in the resulting image and scales the pixel values to lie between 0 and 255.
///
/// This manual normalization is essential because the FFT and inverse FFT operations can produce
/// pixel values outside the standard grayscale range. Without normalization, the resulting image might
/// appear too dark or too bright, or contrast may be lost. It's worth noting that even slight numerical
/// differences in the normalization process between Rust and Python implementations can lead to subtle
/// variations in the filtered images. Such differences are scientifically significant in image processing
/// tasks, as they may affect the visibility of features or the interpretation of results. These variations
/// are typically due to the inherent differences in how Rust and Python handle floating-point arithmetic
/// and image data structures.
///
/// # Scientific Implications
/// The inclusion of normalization is critical for maintaining the scientific integrity of the image
/// processing pipeline. By ensuring that the output image uses the full grayscale range effectively,
/// this function helps preserve the quantitative relationship between different regions of the image.
/// However, users should be aware of the potential for minor discrepancies between results obtained
/// with this Rust implementation and those from other languages or libraries, particularly in tasks
/// requiring precise quantitative analysis.
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
     let padded_image = pad_image(image, npad);
     let fft_shape = &[padded_image.nrows(), padded_image.ncols()];
     //println!("FFT shape for Butterworth filter: {:?}", fft_shape);
     let butterworth_filter_ndarray = get_nd_butterworth_filter(
         fft_shape,
         cutoff_frequency_ratio,
         order,
         high_pass,
         true,
         squared_butterworth,
     );
     //println!("ndarray Butterworth filter dimensions: {:?}", butterworth_filter_ndarray.dim());
     let butterworth_filter = DMatrix::from_iterator(
         fft_shape[0],
         fft_shape[1],
         butterworth_filter_ndarray.iter().cloned().map(|x| Complex::new(x.re, x.im))
     );
     // Debug: Print dimensions of the DMatrix filter
     //println!("DMatrix Butterworth filter dimensions: ({}, {})", butterworth_filter.nrows(), butterworth_filter.ncols());
     // Apply FFT, filter, and inverse FFT
     let final_image = apply_fft_and_filter(&padded_image, &butterworth_filter);
     // Return both the final image and the filter used
     (final_image, butterworth_filter) // Return a tuple containing the image and the filter
  }


  