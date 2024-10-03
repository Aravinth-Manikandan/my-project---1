let model;  // Store the TensorFlow model

// Load TensorFlow model (change path accordingly)
async function loadModel() {
    try {
        document.getElementById('prediction-text').textContent = 'Loading AI model...';
        model = await tf.loadLayersModel('model/model.json');
        document.getElementById('prediction-text').textContent = 'Model loaded. Upload an image!';
    } catch (error) {
        document.getElementById('prediction-text').textContent = 'Error loading model.';
        console.error("Error loading model:", error);
    }
}

// Image upload and display
function loadImage(event) {
    const image = document.getElementById('uploadedImage');
    const file = event.target.files[0];

    if (!file) {
        document.getElementById('result').textContent = "No file selected. Please upload an image.";
        return;
    }

    const reader = new FileReader();

    reader.onload = function(e) {
        image.src = e.target.result;
        image.style.display = 'block';
        document.getElementById('result').textContent = '';
    };

    reader.readAsDataURL(file);
}

// Image preprocessing for model input
function preprocessImage(image) {
    try {
        let tensor = tf.browser.fromPixels(image)
            .resizeNearestNeighbor([224, 224])
            .toFloat();

        tensor = tensor.div(tf.scalar(255));
        return tensor.expandDims(0);
    } catch (error) {
        console.error("Error in image preprocessing:", error);
        document.getElementById('result').textContent = 'Error in image processing.';
    }
}

// Analyze image with the loaded model
async function analyzeImage() {
    try {
        if (!model) {
            document.getElementById('result').textContent = 'Model not loaded yet. Please wait.';
            return;
        }

        const image = document.getElementById('uploadedImage');
        if (!image.src) {
            document.getElementById('result').textContent = 'Please upload an image first.';
            return;
        }

        const tensorImg = preprocessImage(image);
        const predictions = await model.predict(tensorImg).data();
        const result = predictions[0] > 0.5 ? 'Drug usage detected!' : 'No drug usage detected.';
        document.getElementById('result').textContent = `Prediction: ${result}`;
    } catch (error) {
        document.getElementById('result').textContent
