set -e

if ! command -v wasm-pack &> /dev/null; then
    echo "wasm-pack not found, installing..."
    cargo install wasm-pack
fi

echo "Building WASM package..."
wasm-pack build --target web --out-dir pkg

echo "Copying pkg directory to www..."
cp -r pkg www/

mkdir -p www/assets

if [ ! -f www/assets/sample_image.jpg ]; then
    echo "Downloading sample image..."
    curl -o www/assets/sample_image.jpg https://source.unsplash.com/random/800x600/?nature
fi

if [ ! -f www/assets/placeholder.png ]; then
    echo "Creating placeholder image..."
    echo "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=" | base64 -d > www/assets/placeholder.png
fi

if command -v python3 &> /dev/null; then
    echo "Starting a local server with Python..."
    cd www
    python3 -m http.server 8080
elif command -v python &> /dev/null; then
    echo "Starting a local server with Python..."
    cd www
    python -m SimpleHTTPServer 8080
else
    echo "Python not found. Please install Python or use your own web server to serve the files in www/"
    echo "You can use 'cd www && npx serve' if you have Node.js installed."
fi