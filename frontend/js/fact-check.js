/**
 * VisioNova Fact Check Page
 * Handles URL input loading from sessionStorage
 */

document.addEventListener('DOMContentLoaded', function () {
    const urlData = VisioNovaStorage.getFile('url');

    if (urlData) {
        // Populate the URL input field
        const urlInput = document.getElementById('url Input');
        if (urlInput) {
            urlInput.value = urlData.data;
        }
    }
});
