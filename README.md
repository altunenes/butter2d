[![Rust](https://github.com/altunenes/butter2d/actions/workflows/rust.yml/badge.svg)](https://github.com/altunenes/butter2d/actions/workflows/rust.yml) 
[![crates.io](https://img.shields.io/crates/v/butter2d.svg)](https://crates.io/crates/butter2d)
[![DOI](https://zenodo.org/badge/745044483.svg)](https://zenodo.org/doi/10.5281/zenodo.11004855)
[![Deploy WASM](https://github.com/altunenes/butter2d/actions/workflows/deploy-wasm-demo.yml/badge.svg)](https://github.com/altunenes/butter2d/actions/workflows/deploy-wasm-demo.yml)

# butter2d

Pure Rust Implementation of the Butterworth Filter

This crate provides a pure Rust implementation of the Butterworth filter, designed for high-performance spatial frequency filtering of images. It is inspired by and seeks to replicate the functionality of the Butterworth filter as implemented in the popular Python library, [scikit-image](https://github.com/scikit-image/scikit-image/blob/2ac3e141e8d2e31aa0ec10afc3a935396b0618fc/skimage/filters/_fft_based.py#L58-L185).


## ✨ Live WASM Demo

Try the interactive Butterworth filter demo directly in your browser, built with `butter2d` and WebAssembly:

➡️ **[Live Demo Link](https://altunenes.github.io/butter2d/)** ⬅️


## Overview

The Butterworth filter offers a more robust method for applying spatial frequency filters to images compared to traditional FFT/IFFT-based methods. Filters with sharp cutoffs can often lead to the Gibbs phenomenon, where undesirable ringing artifacts appear near edges in the image. This issue is particularly problematic in applications such as EEG experiments (particularly low/mid visual ones that affect P100 amplitudes in the visual cortex) and other scenarios involving low-frequency signals. By providing a smoother transition between the passband and stopband, the Butterworth filter mitigates these effects, making it a preferred choice among vision scientists and image-processing experts.

### Usage

Here's a quick example of applying a high-pass Butterworth filter to an image:
```rust
cargo add butter2d
```

```rust
use image::{GrayImage, open};
use butter2d::butterworth;

fn main() {
    let img = open("path/to/your/image.png").expect("Failed to open image").to_luma8();
    let cutoff_frequency_ratio = 0.1;
    let high_pass = true;
    let order = 2.0;
    let squared_butterworth = false;
    let npad = 0;
    let filtered_img = butterworth(
        &img, 
        cutoff_frequency_ratio, 
        high_pass, 
        order, 
        squared_butterworth, 
        npad
    );
    filtered_img.save("path/to/save/filtered_image.png").expect("Failed to save filtered image");
}
```
