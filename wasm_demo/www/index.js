console.log("Initializing butter2d WASM demo");
import init, { init_panic_hook, FilterParameters, process_image_base64, process_image_base64_color } from './pkg/butter2d_wasm.js';

// DOM elements (will initialize after module loads)
let imageUpload, loadSampleBtn, cutoffFrequency, cutoffValue;
let orderSlider, orderValue, highPassCheckbox, squaredButterworthCheckbox, colorModeCheckbox;
let applyFilterBtn, originalImage, filteredImage;
let imageUrlInput, loadFromUrlBtn;

// Current image as base64
let currentImageBase64 = null;

async function run() {
  try {
    await init();
    init_panic_hook();
    initializeUI();
  } catch (error) {
    console.error("Failed to initialize WASM module:", error);
    document.body.innerHTML = `
      <div style="color: red; padding: 20px; text-align: center;">
        <h2>Failed to load WebAssembly module</h2>
        <p>${error.message}</p>
        <p>Please check that your browser supports WebAssembly and that the module was built correctly.</p>
      </div>
    `;
  }
}

function initializeUI() {
  // Get DOM elements
  imageUpload = document.getElementById('imageUpload');
  loadSampleBtn = document.getElementById('loadSample');
  cutoffFrequency = document.getElementById('cutoffFrequency');
  cutoffValue = document.getElementById('cutoffValue');
  orderSlider = document.getElementById('order');
  orderValue = document.getElementById('orderValue');
  highPassCheckbox = document.getElementById('highPass');
  squaredButterworthCheckbox = document.getElementById('squaredButterworth');
  colorModeCheckbox = document.getElementById('colorMode');
  applyFilterBtn = document.getElementById('applyFilter');
  originalImage = document.getElementById('originalImage');
  filteredImage = document.getElementById('filteredImage');
  imageUrlInput = document.getElementById('imageUrl');
  loadFromUrlBtn = document.getElementById('loadFromUrl');

  cutoffFrequency.addEventListener('input', () => {
    cutoffValue.textContent = cutoffFrequency.value;
  });

  orderSlider.addEventListener('input', () => {
    orderValue.textContent = orderSlider.value;
  });

  imageUpload.addEventListener('change', (event) => {
    const file = event.target.files[0];
    if (file && file.type.match('image.*')) {
      const reader = new FileReader();
      
      reader.onload = (e) => {
        currentImageBase64 = e.target.result;
        originalImage.src = currentImageBase64;
        filteredImage.src = 'assets/placeholder.png';
      };
      
      reader.readAsDataURL(file);
    }
  });

  loadSampleBtn.addEventListener('click', () => {
    fetch('assets/sample_image.jpg')
      .then(response => response.blob())
      .then(blob => {
        const reader = new FileReader();
        reader.onload = (e) => {
          currentImageBase64 = e.target.result;
          originalImage.src = currentImageBase64;
          filteredImage.src = 'assets/placeholder.png';
        };
        reader.readAsDataURL(blob);
      })
      .catch(error => {
        console.error('Error loading sample image:', error);
        alert('Error loading sample image');
      });
  });
  loadFromUrlBtn.addEventListener('click', () => {
    const url = imageUrlInput.value.trim();
    if (!url) {
      alert('Please enter an image URL');
      return;
    }
    loadFromUrlBtn.disabled = true;
    loadFromUrlBtn.textContent = 'Loading...';
    
    fetch(url)
      .then(response => {
        if (!response.ok) {
          throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.blob();
      })
      .then(blob => {
        const reader = new FileReader();
        reader.onload = (e) => {
          currentImageBase64 = e.target.result;
          originalImage.src = currentImageBase64;
          filteredImage.src = 'assets/placeholder.png';
        };
        reader.readAsDataURL(blob);
      })
      .catch(error => {
        console.error('Error loading image from URL:', error);
        alert('Error loading image. The URL might be restricted or invalid.');
      })
      .finally(() => {
        loadFromUrlBtn.disabled = false;
        loadFromUrlBtn.textContent = 'Load from URL';
      });
  });

  applyFilterBtn.addEventListener('click', () => {
    if (!currentImageBase64) {
      alert('Please upload an image, load the sample image, or load an image from URL first.');
      return;
    }
    
    // Show loading state
    applyFilterBtn.disabled = true;
    applyFilterBtn.textContent = 'Processing...';
    
    try {
      // Get filter parameters
      const params = new FilterParameters(
        parseFloat(cutoffFrequency.value),
        highPassCheckbox.checked,
        parseFloat(orderSlider.value),
        squaredButterworthCheckbox.checked
      );
      
      let result;
      if (colorModeCheckbox.checked) {
        result = process_image_base64_color(currentImageBase64, params);
      } else {
        result = process_image_base64(currentImageBase64, params);
      }
      filteredImage.src = result;
    } catch (error) {
      console.error('Error applying filter:', error);
      alert('Error applying filter');
    } finally {
      applyFilterBtn.disabled = false;
      applyFilterBtn.textContent = 'Apply Filter';
    }
  });
  cutoffValue.textContent = cutoffFrequency.value;
  orderValue.textContent = orderSlider.value;
  
  loadSampleBtn.click();
}

run();