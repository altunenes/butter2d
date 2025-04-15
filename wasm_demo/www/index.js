import init, { init_panic_hook, FilterParameters, process_image_base64 } from './pkg/butter2d_wasm.js';

// DOM elements (will initialize after module loads)
let imageUpload, loadSampleBtn, cutoffFrequency, cutoffValue;
let orderSlider, orderValue, highPassCheckbox, squaredButterworthCheckbox;
let applyFilterBtn, originalImage, filteredImage;

// Current image as base64
let currentImageBase64 = null;

async function run() {
  console.log("Initializing WASM module...");
  
  try {
    await init();
    console.log("WASM module initialized successfully");
    
    init_panic_hook();
    console.log("Panic hook initialized");
    
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
  console.log("Initializing UI...");
  
  // Get DOM elements
  imageUpload = document.getElementById('imageUpload');
  loadSampleBtn = document.getElementById('loadSample');
  cutoffFrequency = document.getElementById('cutoffFrequency');
  cutoffValue = document.getElementById('cutoffValue');
  orderSlider = document.getElementById('order');
  orderValue = document.getElementById('orderValue');
  highPassCheckbox = document.getElementById('highPass');
  squaredButterworthCheckbox = document.getElementById('squaredButterworth');
  applyFilterBtn = document.getElementById('applyFilter');
  originalImage = document.getElementById('originalImage');
  filteredImage = document.getElementById('filteredImage');

  cutoffFrequency.addEventListener('input', () => {
    cutoffValue.textContent = cutoffFrequency.value;
  });

  orderSlider.addEventListener('input', () => {
    orderValue.textContent = orderSlider.value;
  });

  imageUpload.addEventListener('change', (event) => {
    console.log("File selected");
    const file = event.target.files[0];
    if (file && file.type.match('image.*')) {
      const reader = new FileReader();
      
      reader.onload = (e) => {
        console.log("File read complete");
        currentImageBase64 = e.target.result;
        originalImage.src = currentImageBase64;
        filteredImage.src = 'assets/placeholder.png';
      };
      
      reader.readAsDataURL(file);
    }
  });

  loadSampleBtn.addEventListener('click', () => {
    console.log("Loading sample image");
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
      });
  });

  applyFilterBtn.addEventListener('click', () => {
    if (!currentImageBase64) {
      alert('Please upload an image or load the sample image first.');
      return;
    }
    
    // Show loading state
    applyFilterBtn.disabled = true;
    applyFilterBtn.textContent = 'Processing...';
    
    try {
      console.log("Creating filter parameters");
      // Get filter parameters
      const params = new FilterParameters(
        parseFloat(cutoffFrequency.value),
        highPassCheckbox.checked,
        parseFloat(orderSlider.value),
        squaredButterworthCheckbox.checked
      );
      
      console.log("Processing image...");
      // Process the image
      const result = process_image_base64(currentImageBase64, params);
      console.log("Image processed successfully");
      filteredImage.src = result;
    } catch (error) {
      console.error('Error applying filter:', error);
      alert('Error applying filter: ' + error.message);
    } finally {
      applyFilterBtn.disabled = false;
      applyFilterBtn.textContent = 'Apply Filter';
    }
  });

  loadSampleBtn.click();
}

run();