use wasm_bindgen::prelude::*;
use image::{GrayImage, ImageBuffer, Rgba, ImageEncoder};
use image::buffer::ConvertBuffer;
use image::codecs::png::PngEncoder;
use butter2d::butterworth;
use base64::{Engine as _, engine::general_purpose};

#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

#[wasm_bindgen]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

#[wasm_bindgen]
pub struct FilterParameters {
    cutoff_frequency_ratio: f64,
    high_pass: bool,
    order: f64,
    squared_butterworth: bool,
}

#[wasm_bindgen]
impl FilterParameters {
    #[wasm_bindgen(constructor)]
    pub fn new(cutoff_frequency_ratio: f64, high_pass: bool, order: f64, squared_butterworth: bool) -> FilterParameters {
        FilterParameters {
            cutoff_frequency_ratio,
            high_pass,
            order,
            squared_butterworth,
        }
    }
}

#[wasm_bindgen]
pub fn apply_filter(image_data_ptr: &[u8], width: u32, height: u32, params: &FilterParameters) -> Result<Vec<u8>, JsValue> {
    // Convert the image_data to GrayImage
    let mut gray_img = GrayImage::new(width, height);
    
    for y in 0..height {
        for x in 0..width {
            let idx = ((y * width + x) * 4) as usize;
            // Simple RGB to grayscale conversion (average method)
            let gray_value = ((image_data_ptr[idx] as u16 + 
                               image_data_ptr[idx + 1] as u16 + 
                               image_data_ptr[idx + 2] as u16) / 3) as u8;
            gray_img.put_pixel(x, y, image::Luma([gray_value]));
        }
    }
    
    // Apply the Butterworth filter
    let (filtered_img, _) = butterworth(
        &gray_img,
        params.cutoff_frequency_ratio,
        params.high_pass,
        params.order,
        params.squared_butterworth,
        0, // No padding
    );
    
    // Convert filtered grayscale image back to RGBA
    let mut rgba_data = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..filtered_img.height() {
        for x in 0..filtered_img.width() {
            let pixel = filtered_img.get_pixel(x, y);
            rgba_data.push(pixel[0]); // R
            rgba_data.push(pixel[0]); // G
            rgba_data.push(pixel[0]); // B
            rgba_data.push(255);      // A (fully opaque)
        }
    }
    
    Ok(rgba_data)
}
#[wasm_bindgen]
pub fn process_image_base64(image_base64: &str, params: &FilterParameters) -> Result<String, JsValue> {
    let base64_str = if image_base64.starts_with("data:image") {
        image_base64.split(",").nth(1).unwrap_or(image_base64)
    } else {
        image_base64
    };
    
    // Decode base64 string to bytes
    let image_data = match general_purpose::STANDARD.decode(base64_str) {
        Ok(data) => data,
        Err(e) => return Err(JsValue::from_str(&format!("Failed to decode base64: {}", e))),
    };
    
    // Load the image from bytes
    let img = match image::load_from_memory(&image_data) {
        Ok(img) => img,
        Err(e) => return Err(JsValue::from_str(&format!("Failed to load image: {}", e))),
    };
    
    // Convert to grayscale
    let gray_img = img.to_luma8();
    
    // Apply the Butterworth filter
    let (filtered_img, _) = butterworth(
        &gray_img,
        params.cutoff_frequency_ratio,
        params.high_pass,
        params.order,
        params.squared_butterworth,
        0, // No padding
    );
    
    // Convert back to RGB for display
    let width = filtered_img.width();
    let height = filtered_img.height();
    let mut rgb_img: ImageBuffer<Rgba<u8>, Vec<u8>> = ImageBuffer::new(width, height);
    
    for y in 0..height {
        for x in 0..width {
            let pixel = filtered_img.get_pixel(x, y);
            rgb_img.put_pixel(x, y, Rgba([pixel[0], pixel[0], pixel[0], 255]));
        }
    }
    
    // Encode the filtered image to PNG and then to base64
    let mut buffer = Vec::new();
    let rgb_img_converted: image::RgbImage = rgb_img.convert();
    if let Err(e) = PngEncoder::new(&mut buffer).write_image(
        rgb_img_converted.as_raw(),
        rgb_img_converted.width(),
        rgb_img_converted.height(),
        image::ColorType::Rgb8.into()
    ) {
        return Err(JsValue::from_str(&format!("Failed to encode image: {}", e)));
    }
    
    let base64_output = general_purpose::STANDARD.encode(&buffer);
    Ok(format!("data:image/png;base64,{}", base64_output))
}
#[wasm_bindgen]
pub fn process_image_base64_color(image_base64: &str, params: &FilterParameters) -> Result<String, JsValue> {
    let base64_str = if image_base64.starts_with("data:image") {
        image_base64.split(",").nth(1).unwrap_or(image_base64)
    } else {
        image_base64
    };
    
    let image_data = match general_purpose::STANDARD.decode(base64_str) {
        Ok(data) => data,
        Err(e) => return Err(JsValue::from_str(&format!("Failed to decode base64: {}", e))),
    };
    
    let img = match image::load_from_memory(&image_data) {
        Ok(img) => img,
        Err(e) => return Err(JsValue::from_str(&format!("Failed to load image: {}", e))),
    };
    // Convert to RGB (preserving color)
    let rgb_img = img.to_rgb8();
    let (filtered_img, _) = butter2d::butterworth_color(
        &rgb_img,
        params.cutoff_frequency_ratio,
        params.high_pass,
        params.order,
        params.squared_butterworth,
        0,
    );
    
    let mut buffer = Vec::new();
    if let Err(e) = PngEncoder::new(&mut buffer).write_image(
        filtered_img.as_raw(),
        filtered_img.width(),
        filtered_img.height(),
        image::ColorType::Rgb8.into()
    ) {
        return Err(JsValue::from_str(&format!("Failed to encode image: {}", e)));
    }
    
    let base64_output = general_purpose::STANDARD.encode(&buffer);
    Ok(format!("data:image/png;base64,{}", base64_output))
}