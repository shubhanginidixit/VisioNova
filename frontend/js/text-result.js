/**
 * VisioNova Text Result Page
 * Handles dynamic text content loading from sessionStorage
 */

document.addEventListener('DOMContentLoaded', function () {
    const textData = VisioNovaStorage.getFile('text');

    if (textData) {
        // Update page title with filename
        const pageTitle = document.getElementById('pageTitle');
        if (pageTitle) {
            pageTitle.textContent = 'Analysis: ' + textData.fileName;
        }

        // Display the uploaded text
        const placeholder = document.getElementById('noTextPlaceholder');
        const uploadedText = document.getElementById('uploadedText');

        // Split text into paragraphs and format
        const paragraphs = textData.data.split('\n\n').filter(p => p.trim());
        let formattedHtml = '';
        paragraphs.forEach(p => {
            formattedHtml += `<p class="mb-4">${p.replace(/\n/g, '<br>')}</p>`;
        });

        if (!formattedHtml) {
            formattedHtml = `<p class="mb-4">${textData.data.replace(/\n/g, '<br>')}</p>`;
        }

        uploadedText.innerHTML = formattedHtml;
        placeholder.classList.add('hidden');
        uploadedText.classList.remove('hidden');
    }
});
