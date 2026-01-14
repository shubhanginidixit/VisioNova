/**
 * VisioNova Audio Result Page
 * Handles dynamic audio content loading from sessionStorage
 */

document.addEventListener('DOMContentLoaded', function () {
    const audioData = VisioNovaStorage.getFile('audio');

    if (audioData) {
        // Update page title with filename
        const pageTitle = document.getElementById('pageTitle');
        if (pageTitle) {
            pageTitle.textContent = audioData.fileName;
        }

        // Display the uploaded audio
        const placeholder = document.getElementById('noAudioPlaceholder');
        const uploadedAudio = document.getElementById('uploadedAudio');
        const spectrogramBg = document.getElementById('spectrogramBg');

        if (audioData.mimeType === 'url') {
            uploadedAudio.src = audioData.data;
        } else {
            uploadedAudio.src = audioData.data;
        }

        placeholder.classList.add('hidden');
        uploadedAudio.classList.remove('hidden');
        spectrogramBg.classList.remove('hidden');
    }
});
