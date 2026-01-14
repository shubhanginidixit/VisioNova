/**
 * VisioNova Result Page - Image Analysis
 * Handles dynamic content loading from sessionStorage
 */

document.addEventListener('DOMContentLoaded', function () {
    // Get uploaded image data from storage
    const imageData = VisioNovaStorage.getFile('image');

    if (imageData) {
        // Update page title with filename
        const pageTitle = document.getElementById('pageTitle');
        if (pageTitle) {
            pageTitle.textContent = 'Analysis: ' + imageData.fileName;
        }

        // Update timestamp
        const analysisTime = document.getElementById('analysisTime');
        if (analysisTime) {
            const date = new Date(imageData.timestamp);
            analysisTime.textContent = date.toLocaleDateString() + ' at ' + date.toLocaleTimeString();
        }

        // Display the uploaded image
        const imageContainer = document.getElementById('imageContainer');
        const placeholder = document.getElementById('noImagePlaceholder');
        const uploadedImage = document.getElementById('uploadedImage');

        if (imageData.mimeType === 'url') {
            // It's a URL, set as image src directly
            uploadedImage.src = imageData.data;
        } else {
            // It's base64 data
            uploadedImage.src = imageData.data;
        }

        placeholder.classList.add('hidden');
        uploadedImage.classList.remove('hidden');
    } else {
        // No image uploaded - show placeholder (default state)
        const analysisTime = document.getElementById('analysisTime');
        if (analysisTime) {
            analysisTime.textContent = 'No analysis performed';
        }
    }
});
