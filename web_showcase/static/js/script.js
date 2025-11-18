// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const uploadBtn = document.getElementById('uploadBtn');
const previewSection = document.getElementById('previewSection');
const previewImage = document.getElementById('previewImage');
const previewVideo = document.getElementById('previewVideo');
const analyzeBtn = document.getElementById('analyzeBtn');
const resultsSection = document.getElementById('resultsSection');
const predictionBadge = document.getElementById('predictionBadge');
const confidenceScore = document.getElementById('confidenceScore');

// File handling
let selectedFile = null;

// Upload area event listeners
uploadArea.addEventListener('click', () => fileInput.click());
uploadBtn.addEventListener('click', () => fileInput.click());

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileSelection(files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileSelection(e.target.files[0]);
    }
});

// File selection handler
function handleFileSelection(file) {
    selectedFile = file;

    // Show preview
    previewSection.style.display = 'block';
    resultsSection.style.display = 'none';

    if (file.type.startsWith('image/')) {
        // Image preview
        previewImage.style.display = 'block';
        previewVideo.style.display = 'none';

        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
        };
        reader.readAsDataURL(file);
    } else if (file.type.startsWith('video/')) {
        // Video preview
        previewImage.style.display = 'none';
        previewVideo.style.display = 'block';

        const videoURL = URL.createObjectURL(file);
        previewVideo.src = videoURL;
    }

    // Update button text
    analyzeBtn.innerHTML = '<i class="fas fa-search"></i> Analyze Media';
    analyzeBtn.disabled = false;
}

// Analyze button click
analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    // Show loading state
    analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
    analyzeBtn.disabled = true;

    predictionBadge.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
    predictionBadge.className = 'prediction-badge analyzing';

    resultsSection.style.display = 'block';

    try {
        // Create form data
        const formData = new FormData();
        formData.append('file', selectedFile);

        // Send request
        const response = await fetch('/api/detect', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            displayResults(data.result);
        } else {
            showError(data.error);
        }
    } catch (error) {
        showError('Network error occurred');
        console.error('Detection error:', error);
    } finally {
        analyzeBtn.innerHTML = '<i class="fas fa-search"></i> Analyze Media';
        analyzeBtn.disabled = false;
    }
});

// Display results
function displayResults(result) {
    const { prediction, confidence, method, risk_level, inference_time } = result;

    // Update prediction badge
    const icon = prediction === 'REAL' ? 'fa-check-circle' : 'fa-exclamation-triangle';
    const colorClass = prediction === 'REAL' ? 'real' : 'fake';

    predictionBadge.innerHTML = `
        <i class="fas ${icon}"></i> ${prediction}
    `;
    predictionBadge.className = `prediction-badge ${colorClass}`;

    // Update confidence score
    confidenceScore.textContent = `${(confidence * 100).toFixed(1)}%`;

    // Update details
    document.getElementById('methodUsed').textContent = method || 'Hybrid Model';
    document.getElementById('riskLevel').textContent = risk_level || 'UNKNOWN';
    document.getElementById('processingTime').textContent = `${(inference_time * 1000).toFixed(0)}ms`;

    // Update branch analysis if available
    if (result.spatial_confidence !== undefined) {
        document.getElementById('branchAnalysis').style.display = 'block';

        updateBranchBar('spatialBar', 'spatialValue', result.spatial_confidence);
        updateBranchBar('frequencyBar', 'frequencyValue', result.frequency_confidence);
        updateBranchBar('textureBar', 'textureValue', result.texture_confidence);
    } else {
        document.getElementById('branchAnalysis').style.display = 'none';
    }
}

function updateBranchBar(barId, valueId, confidence) {
    const bar = document.getElementById(barId);
    const value = document.getElementById(valueId);

    const percentage = (confidence * 100).toFixed(1);
    bar.style.width = `${percentage}%`;
    value.textContent = `${percentage}%`;
}

// Error handling
function showError(message) {
    predictionBadge.innerHTML = '<i class="fas fa-exclamation-circle"></i> Error';
    predictionBadge.className = 'prediction-badge error';
    confidenceScore.textContent = '--';

    alert(`Analysis failed: ${message}`);
}

// Load metrics on page load
document.addEventListener('DOMContentLoaded', loadMetrics);

async function loadMetrics() {
    try {
        const response = await fetch('/api/metrics');
        const metrics = await response.json();

        // Update metric values
        document.getElementById('accuracyValue').textContent =
            metrics.best_val_acc ? `${(metrics.best_val_acc * 100).toFixed(1)}%` : '--';
        document.getElementById('aucValue').textContent =
            metrics.best_val_auc ? metrics.best_val_auc.toFixed(4) : '--';
        document.getElementById('f1Value').textContent =
            metrics.best_val_f1 ? `${(metrics.best_val_f1 * 100).toFixed(1)}%` : '--';
        document.getElementById('recallValue').textContent =
            metrics.best_val_recall ? `${(metrics.best_val_recall * 100).toFixed(1)}%` : '--';
        document.getElementById('precisionValue').textContent =
            metrics.best_val_precision ? `${(metrics.best_val_precision * 100).toFixed(1)}%` : '--';
        document.getElementById('epochsValue').textContent =
            metrics.epochs_trained || '--';

    } catch (error) {
        console.error('Failed to load metrics:', error);
    }
}

// Reset upload when clicking upload area again
uploadArea.addEventListener('click', () => {
    if (selectedFile) {
        // Reset to initial state
        selectedFile = null;
        previewSection.style.display = 'none';
        resultsSection.style.display = 'none';

        // Clear video object URL to prevent memory leaks
        if (previewVideo.src) {
            URL.revokeObjectURL(previewVideo.src);
            previewVideo.src = '';
        }
    }
});
