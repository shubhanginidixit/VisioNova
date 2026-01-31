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

    // Text file input - handles TXT (as text) and PDF/DOCX (as File for backend parsing)
    document.getElementById('textFileInput').addEventListener('change', async function (e) {
        if (e.target.files.length > 0) {
            const file = e.target.files[0];
            const fileName = file.name.toLowerCase();
            const isPdfOrDocx = fileName.endsWith('.pdf') || fileName.endsWith('.docx') || fileName.endsWith('.doc');

            if (isPdfOrDocx) {
                // For PDF/DOCX, store the File object directly for FormData upload
                VisioNovaStorage.saveDocumentFile(file, file.name);
                uploadedFiles.text = {
                    isDocument: true,
                    fileName: file.name,
                    mimeType: file.type || 'application/octet-stream'
                };
                showUploadSuccess('text', file.name);
                const textarea = document.querySelector('#text-upload textarea');
                if (textarea) {
                    textarea.value = `[Document: ${file.name}]\n\nThis document will be sent to the backend for text extraction and AI analysis.`;
                    textarea.disabled = true;
                }
            } else {
                // For plain text files, read as text
                const reader = new FileReader();
                reader.onload = function (event) {
                    const textarea = document.querySelector('#text-upload textarea');
                    if (textarea) {
                        textarea.value = event.target.result;
                        textarea.disabled = false;
                    }
                    uploadedFiles.text = {
                        data: event.target.result,
                        fileName: file.name,
                        mimeType: 'text/plain',
                        isDocument: false
                    };
                    showUploadSuccess('text', file.name);
                };
                reader.readAsText(file);
            }
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
        showPreview(type, dataURL, file.type, file.name);
    } catch (error) {
        console.error('Error reading file:', error);
        alert('Error reading file. Please try again.');
    }
}

// Show upload success indicator
function showUploadSuccess(type, fileName) {
    // For image type, we now use the preview state instead of floating indicator
    if (type === 'image') {
        return; // Handled by showPreview
    }
    
    const uploadArea = document.getElementById(type + '-upload');
    if (!uploadArea) return;

    // Find or create success indicator for non-image types
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
function showPreview(type, dataURL, mimeType, fileName) {
    if (type === 'image') {
        // Use the new state-switching UI for images
        const defaultState = document.getElementById('image-upload-default');
        const previewState = document.getElementById('image-preview-state');
        const previewImg = document.getElementById('image-preview-img');
        const fileNameEl = document.getElementById('image-file-name');
        
        if (defaultState && previewState && previewImg) {
            // Set preview image
            previewImg.src = dataURL;
            
            // Set filename
            if (fileNameEl && fileName) {
                fileNameEl.textContent = fileName;
            }
            
            // Switch states with fade effect
            defaultState.classList.add('opacity-0');
            setTimeout(() => {
                defaultState.classList.add('hidden');
                previewState.classList.remove('hidden');
                setTimeout(() => {
                    previewState.classList.remove('opacity-0');
                }, 10);
            }, 150);
        }
        return;
    }
    
    // Handle video/audio with original method
    const uploadArea = document.getElementById(type + '-upload');
    if (!uploadArea) return;

    const dropZone = uploadArea.querySelector('.border-dashed');
    if (!dropZone) return;

    // Clear previous preview
    const existingPreview = dropZone.querySelector('.file-preview');
    if (existingPreview) existingPreview.remove();

    // Create preview based on type
    let previewHTML = '';
    if (type === 'video') {
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

// Clear image preview and reset to default state
function clearImagePreview() {
    const defaultState = document.getElementById('image-upload-default');
    const previewState = document.getElementById('image-preview-state');
    const previewImg = document.getElementById('image-preview-img');
    
    if (defaultState && previewState) {
        // Switch back to default state with fade
        previewState.classList.add('opacity-0');
        setTimeout(() => {
            previewState.classList.add('hidden');
            if (previewImg) previewImg.src = '';
            defaultState.classList.remove('hidden');
            setTimeout(() => {
                defaultState.classList.remove('opacity-0');
            }, 10);
        }, 150);
    }
    
    // Clear stored file data
    uploadedFiles.image = null;
    
    // Clear URL input
    const urlInput = document.getElementById('imageUrlInput');
    if (urlInput) urlInput.value = '';
    
    // Reset file input
    const fileInput = document.getElementById('imageFileInput');
    if (fileInput) fileInput.value = '';
}

// Setup clear button handler
function setupClearButton() {
    const clearBtn = document.getElementById('clearImageBtn');
    if (clearBtn) {
        clearBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            clearImagePreview();
        });
    }
    
    // Also setup browse link
    const browseLink = document.getElementById('imageBrowseLink');
    if (browseLink) {
        browseLink.addEventListener('click', (e) => {
            e.stopPropagation();
            document.getElementById('imageFileInput').click();
        });
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

// Setup clipboard paste for images
function setupClipboardPaste() {
    // Paste button click handler
    const pasteBtn = document.getElementById('pasteImageBtn');
    if (pasteBtn) {
        pasteBtn.addEventListener('click', async () => {
            await pasteImageFromClipboard();
        });
    }

    // Global paste handler (Ctrl+V)
    document.addEventListener('paste', async (e) => {
        // Only handle paste when image tab is active
        if (activeTab !== 'image') return;
        
        // Don't intercept paste in input fields
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
        
        await handlePasteEvent(e);
    });

    // Also allow paste when image drop zone is focused
    const dropZone = document.getElementById('image-drop-zone');
    if (dropZone) {
        dropZone.addEventListener('paste', async (e) => {
            e.preventDefault();
            await handlePasteEvent(e);
        });

        // Allow keyboard focus for accessibility
        dropZone.addEventListener('keydown', async (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                document.getElementById('imageFileInput').click();
            }
        });
    }
}

// Handle paste event
async function handlePasteEvent(e) {
    const clipboardData = e.clipboardData || window.clipboardData;
    if (!clipboardData) return;

    // Check for image in clipboard
    const items = clipboardData.items;
    for (let i = 0; i < items.length; i++) {
        if (items[i].type.indexOf('image') !== -1) {
            e.preventDefault();
            const file = items[i].getAsFile();
            if (file) {
                await handleFileUpload('image', file);
                showPasteSuccess();
            }
            return;
        }
    }

    // Check for image URL in text
    const text = clipboardData.getData('text');
    if (text && isImageUrl(text)) {
        e.preventDefault();
        const urlInput = document.querySelector('#image-upload input[type="text"]');
        if (urlInput) {
            urlInput.value = text;
            showPasteSuccess('URL pasted');
        }
    }
}

// Paste image from clipboard using Clipboard API
async function pasteImageFromClipboard() {
    try {
        // Check if Clipboard API is available
        if (!navigator.clipboard || !navigator.clipboard.read) {
            // Fallback: prompt user to use Ctrl+V
            showPasteError('Please use Ctrl+V to paste, or grant clipboard permission');
            return;
        }

        const clipboardItems = await navigator.clipboard.read();
        
        for (const item of clipboardItems) {
            // Check for image types
            const imageTypes = item.types.filter(type => type.startsWith('image/'));
            
            if (imageTypes.length > 0) {
                const blob = await item.getType(imageTypes[0]);
                const file = new File([blob], `pasted-image-${Date.now()}.png`, { type: imageTypes[0] });
                await handleFileUpload('image', file);
                showPasteSuccess();
                return;
            }

            // Check for text that might be an image URL
            if (item.types.includes('text/plain')) {
                const blob = await item.getType('text/plain');
                const text = await blob.text();
                if (isImageUrl(text)) {
                    const urlInput = document.getElementById('imageUrlInput');
                    if (urlInput) {
                        urlInput.value = text;
                        showPasteSuccess('URL pasted');
                    }
                    return;
                }
            }
        }

        showPasteError('No image found in clipboard');
    } catch (error) {
        console.error('Clipboard read error:', error);
        if (error.name === 'NotAllowedError') {
            showPasteError('Clipboard access denied. Please use Ctrl+V to paste');
        } else {
            showPasteError('Could not read clipboard. Try Ctrl+V');
        }
    }
}

// Check if a string is an image URL
function isImageUrl(url) {
    if (!url) return false;
    try {
        const urlObj = new URL(url);
        const path = urlObj.pathname.toLowerCase();
        return /\.(jpg|jpeg|png|gif|webp|bmp|svg)$/i.test(path) ||
               url.includes('images') ||
               url.includes('imgur') ||
               url.includes('i.redd.it');
    } catch {
        return false;
    }
}

// Show paste success indicator
function showPasteSuccess(message = 'Image pasted!') {
    const uploadArea = document.getElementById('image-upload');
    if (!uploadArea) return;

    // Create toast notification
    let toast = document.getElementById('paste-toast');
    if (!toast) {
        toast = document.createElement('div');
        toast.id = 'paste-toast';
        toast.className = 'fixed bottom-6 right-6 flex items-center gap-2 px-4 py-3 bg-green-500/90 backdrop-blur-sm rounded-lg text-white text-sm font-medium shadow-lg z-50 transform translate-y-20 opacity-0 transition-all duration-300';
        document.body.appendChild(toast);
    }

    toast.innerHTML = `
        <span class="material-symbols-outlined">check_circle</span>
        <span>${message}</span>
    `;

    // Animate in
    requestAnimationFrame(() => {
        toast.classList.remove('translate-y-20', 'opacity-0');
    });

    // Remove after delay
    setTimeout(() => {
        toast.classList.add('translate-y-20', 'opacity-0');
    }, 2500);
}

// Show paste error indicator
function showPasteError(message) {
    let toast = document.getElementById('paste-toast');
    if (!toast) {
        toast = document.createElement('div');
        toast.id = 'paste-toast';
        toast.className = 'fixed bottom-6 right-6 flex items-center gap-2 px-4 py-3 bg-red-500/90 backdrop-blur-sm rounded-lg text-white text-sm font-medium shadow-lg z-50 transform translate-y-20 opacity-0 transition-all duration-300';
        document.body.appendChild(toast);
    }

    toast.className = toast.className.replace('bg-green-500/90', 'bg-red-500/90');
    toast.innerHTML = `
        <span class="material-symbols-outlined">error</span>
        <span>${message}</span>
    `;

    requestAnimationFrame(() => {
        toast.classList.remove('translate-y-20', 'opacity-0');
    });

    setTimeout(() => {
        toast.classList.add('translate-y-20', 'opacity-0');
        // Reset color
        setTimeout(() => {
            toast.className = toast.className.replace('bg-red-500/90', 'bg-green-500/90');
        }, 300);
    }, 3000);
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
        if (uploadedFiles.text && uploadedFiles.text.isDocument) {
            // PDF/DOCX file - already stored via saveDocumentFile, just set the flag
            sessionStorage.setItem('visioNova_isDocument', 'true');
            sessionStorage.setItem('visioNova_documentFileName', uploadedFiles.text.fileName);
        } else if (textarea && textarea.value.trim() && !textarea.disabled) {
            VisioNovaStorage.saveFile('text', textarea.value.trim(), 'Pasted Text', 'text/plain');
            sessionStorage.removeItem('visioNova_isDocument');
        } else if (uploadedFiles.text) {
            VisioNovaStorage.saveFile('text', uploadedFiles.text.data, uploadedFiles.text.fileName, 'text/plain');
            sessionStorage.removeItem('visioNova_isDocument');
        }
    } else if (activeTab === 'url') {
        const urlInput = document.querySelector('#url-upload input[type="url"]');
        if (urlInput && urlInput.value.trim()) {
            VisioNovaStorage.saveFile('url', urlInput.value.trim(), 'Fact Check URL', 'text/uri');
        }
    }

    // Navigate to analysis dashboard for processing
    const activeTabElement = document.querySelector('.media-tab.active-tab');
    if (activeTabElement) {
        const resultPage = activeTabElement.getAttribute('data-result');
        if (resultPage) {
            // Redirect to AnalysisDashboard with target
            window.location.href = `AnalysisDashboard.html?analyzing=true&media=${activeTab}&next=${encodeURIComponent(resultPage)}`;
        }
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function () {
    setupBrowseLinks();
    setupFileInputs();
    setupDragAndDrop();
    setupClipboardPaste();
    setupClearButton();
});
