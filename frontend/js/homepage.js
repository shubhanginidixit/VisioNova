/**
 * VisioNova Homepage - Tab Switching and File Upload
 */

// Track currently active tab and uploaded files
let activeTab = 'image';
let uploadedFiles = {
    image: null,
    video: null,
    audio: null,
    text: null,
    url: null
};

// Tab switching logic
document.querySelectorAll('.media-tab').forEach(tab => {
    tab.addEventListener('click', function () {
        const targetTab = this.getAttribute('data-tab');
        activeTab = targetTab;

        // Remove active state from all tabs
        document.querySelectorAll('.media-tab').forEach(t => {
            t.classList.remove('active-tab');
            t.classList.add('inactive-tab');
        });

        // Add active state to clicked tab
        this.classList.remove('inactive-tab');
        this.classList.add('active-tab');

        // Hide all upload content sections
        document.querySelectorAll('.upload-content').forEach(content => {
            content.classList.add('hidden');
        });

        // Show the corresponding upload content
        const targetContent = document.getElementById(targetTab + '-upload');
        if (targetContent) {
            targetContent.classList.remove('hidden');
        }
    });
});

// Setup file input triggers for "browse files" links
function setupBrowseLinks() {
    // Image browse
    const imageBrowse = document.querySelector('#image-upload .text-primary.cursor-pointer');
    if (imageBrowse) {
        imageBrowse.addEventListener('click', () => document.getElementById('imageFileInput').click());
    }

    // Video browse
    const videoBrowse = document.querySelector('#video-upload .text-purple-400.cursor-pointer');
    if (videoBrowse) {
        videoBrowse.addEventListener('click', () => document.getElementById('videoFileInput').click());
    }

    // Audio browse
    const audioBrowse = document.querySelector('#audio-upload .text-green-400.cursor-pointer');
    if (audioBrowse) {
        audioBrowse.addEventListener('click', () => document.getElementById('audioFileInput').click());
    }

    // Text file browse button
    const textBrowseBtn = document.querySelector('#text-upload button.text-amber-400');
    if (textBrowseBtn) {
        textBrowseBtn.addEventListener('click', () => document.getElementById('textFileInput').click());
    }
}

// Handle file selection
function setupFileInputs() {
    // Image file input
    document.getElementById('imageFileInput').addEventListener('change', async function (e) {
        if (e.target.files.length > 0) {
            await handleFileUpload('image', e.target.files[0]);
        }
    });

    // Video file input
    document.getElementById('videoFileInput').addEventListener('change', async function (e) {
        if (e.target.files.length > 0) {
            await handleFileUpload('video', e.target.files[0]);
        }
    });

    // Audio file input
    document.getElementById('audioFileInput').addEventListener('change', async function (e) {
        if (e.target.files.length > 0) {
            await handleFileUpload('audio', e.target.files[0]);
        }
    });

    // Text file input
    document.getElementById('textFileInput').addEventListener('change', async function (e) {
        if (e.target.files.length > 0) {
            const file = e.target.files[0];
            const reader = new FileReader();
            reader.onload = function (event) {
                const textarea = document.querySelector('#text-upload textarea');
                if (textarea) {
                    textarea.value = event.target.result;
                }
                uploadedFiles.text = {
                    data: event.target.result,
                    fileName: file.name,
                    mimeType: 'text/plain'
                };
                showUploadSuccess('text', file.name);
            };
            reader.readAsText(file);
        }
    });
}

// Handle file upload for image/video/audio
async function handleFileUpload(type, file) {
    try {
        const dataURL = await VisioNovaStorage.readFileAsDataURL(file);
        uploadedFiles[type] = {
            data: dataURL,
            fileName: file.name,
            mimeType: file.type
        };
        showUploadSuccess(type, file.name);
        showPreview(type, dataURL, file.type);
    } catch (error) {
        console.error('Error reading file:', error);
        alert('Error reading file. Please try again.');
    }
}

// Show upload success indicator
function showUploadSuccess(type, fileName) {
    const uploadArea = document.getElementById(type + '-upload');
    if (!uploadArea) return;

    // Find or create success indicator
    let indicator = uploadArea.querySelector('.upload-success-indicator');
    if (!indicator) {
        indicator = document.createElement('div');
        indicator.className = 'upload-success-indicator absolute top-4 right-4 flex items-center gap-2 px-3 py-2 bg-green-500/20 border border-green-500/30 rounded-lg text-green-400 text-sm z-30';
        indicator.innerHTML = `
            <span class="material-symbols-outlined text-lg">check_circle</span>
            <span class="file-name truncate max-w-[200px]"></span>
        `;
        uploadArea.appendChild(indicator);
    }
    indicator.querySelector('.file-name').textContent = fileName;
}

// Show preview for uploaded file
function showPreview(type, dataURL, mimeType) {
    const uploadArea = document.getElementById(type + '-upload');
    if (!uploadArea) return;

    const dropZone = uploadArea.querySelector('.border-dashed');
    if (!dropZone) return;

    // Clear previous preview
    const existingPreview = dropZone.querySelector('.file-preview');
    if (existingPreview) existingPreview.remove();

    // Create preview based on type
    let previewHTML = '';
    if (type === 'image') {
        previewHTML = `<img src="${dataURL}" alt="Preview" class="max-h-48 max-w-full rounded-lg object-contain" />`;
    } else if (type === 'video') {
        previewHTML = `<video src="${dataURL}" class="max-h-48 max-w-full rounded-lg" controls></video>`;
    } else if (type === 'audio') {
        previewHTML = `<audio src="${dataURL}" class="w-full max-w-md" controls></audio>`;
    }

    if (previewHTML) {
        const previewDiv = document.createElement('div');
        previewDiv.className = 'file-preview flex flex-col items-center gap-2 mt-4';
        previewDiv.innerHTML = previewHTML;
        dropZone.appendChild(previewDiv);
    }
}

// Setup drag and drop for upload zones
function setupDragAndDrop() {
    const dropZones = {
        'image-upload': { input: 'imageFileInput', type: 'image' },
        'video-upload': { input: 'videoFileInput', type: 'video' },
        'audio-upload': { input: 'audioFileInput', type: 'audio' }
    };

    Object.keys(dropZones).forEach(zoneId => {
        const zone = document.getElementById(zoneId);
        if (!zone) return;

        const dropArea = zone.querySelector('.border-dashed');
        if (!dropArea) return;

        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropArea.addEventListener(eventName, e => {
                e.preventDefault();
                e.stopPropagation();
            });
        });

        ['dragenter', 'dragover'].forEach(eventName => {
            dropArea.addEventListener(eventName, () => {
                dropArea.classList.add('border-primary', 'bg-primary/10');
            });
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropArea.addEventListener(eventName, () => {
                dropArea.classList.remove('border-primary', 'bg-primary/10');
            });
        });

        dropArea.addEventListener('drop', async e => {
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                await handleFileUpload(dropZones[zoneId].type, files[0]);
            }
        });
    });
}

// Navigate to the result page based on active tab
function navigateToResult() {
    // Save data to storage before navigating
    if (activeTab === 'image') {
        if (uploadedFiles.image) {
            VisioNovaStorage.saveFile('image', uploadedFiles.image.data, uploadedFiles.image.fileName, uploadedFiles.image.mimeType);
        } else {
            // Check for URL input
            const urlInput = document.querySelector('#image-upload input[type="text"]');
            if (urlInput && urlInput.value.trim()) {
                VisioNovaStorage.saveFile('image', urlInput.value.trim(), 'URL Image', 'url');
            }
        }
    } else if (activeTab === 'video') {
        if (uploadedFiles.video) {
            VisioNovaStorage.saveFile('video', uploadedFiles.video.data, uploadedFiles.video.fileName, uploadedFiles.video.mimeType);
        } else {
            const urlInput = document.querySelector('#video-upload input[type="text"]');
            if (urlInput && urlInput.value.trim()) {
                VisioNovaStorage.saveFile('video', urlInput.value.trim(), 'URL Video', 'url');
            }
        }
    } else if (activeTab === 'audio') {
        if (uploadedFiles.audio) {
            VisioNovaStorage.saveFile('audio', uploadedFiles.audio.data, uploadedFiles.audio.fileName, uploadedFiles.audio.mimeType);
        } else {
            const urlInput = document.querySelector('#audio-upload input[type="text"]');
            if (urlInput && urlInput.value.trim()) {
                VisioNovaStorage.saveFile('audio', urlInput.value.trim(), 'URL Audio', 'url');
            }
        }
    } else if (activeTab === 'text') {
        const textarea = document.querySelector('#text-upload textarea');
        if (textarea && textarea.value.trim()) {
            VisioNovaStorage.saveFile('text', textarea.value.trim(), 'Pasted Text', 'text/plain');
        } else if (uploadedFiles.text) {
            VisioNovaStorage.saveFile('text', uploadedFiles.text.data, uploadedFiles.text.fileName, 'text/plain');
        }
    } else if (activeTab === 'url') {
        const urlInput = document.querySelector('#url-upload input[type="url"]');
        if (urlInput && urlInput.value.trim()) {
            VisioNovaStorage.saveFile('url', urlInput.value.trim(), 'Fact Check URL', 'text/uri');
        }
    }

    // Navigate to result page
    const activeTabElement = document.querySelector('.media-tab.active-tab');
    if (activeTabElement) {
        const resultPage = activeTabElement.getAttribute('data-result');
        if (resultPage) {
            window.location.href = resultPage;
        }
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function () {
    setupBrowseLinks();
    setupFileInputs();
    setupDragAndDrop();
});
