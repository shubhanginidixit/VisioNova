/**
 * VisioNova Video Result Page
 * Handles dynamic video content loading from sessionStorage
 */

document.addEventListener('DOMContentLoaded', function () {
    const videoData = VisioNovaStorage.getFile('video');

    if (videoData) {
        // Update page title with filename
        const pageTitle = document.getElementById('pageTitle');
        if (pageTitle) {
            pageTitle.textContent = 'Analysis: ' + videoData.fileName;
        }

        // Display the uploaded video
        const placeholder = document.getElementById('noVideoPlaceholder');
        const uploadedVideo = document.getElementById('uploadedVideo');

        if (videoData.mimeType === 'url') {
            uploadedVideo.src = videoData.data;
        } else {
            uploadedVideo.src = videoData.data;
        }

        placeholder.classList.add('hidden');
        uploadedVideo.classList.remove('hidden');
    }
});
