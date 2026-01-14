/**
 * VisioNova Storage Utilities
 * Handles file storage/retrieval using sessionStorage for passing data between pages
 */

const VisioNovaStorage = {
    // Storage keys for each media type
    KEYS: {
        image: 'visioNova_image',
        video: 'visioNova_video',
        audio: 'visioNova_audio',
        text: 'visioNova_text',
        url: 'visioNova_url'
    },

    /**
     * Save uploaded file data to sessionStorage
     * @param {string} type - Media type (image, video, audio, text, url)
     * @param {string} data - Base64 encoded file data or text/url content
     * @param {string} fileName - Original file name
     * @param {string} mimeType - MIME type of the file
     */
    saveFile: function (type, data, fileName, mimeType) {
        const key = this.KEYS[type];
        if (!key) {
            console.error('Invalid file type:', type);
            return false;
        }

        const fileData = {
            data: data,
            fileName: fileName || 'Untitled',
            mimeType: mimeType || 'application/octet-stream',
            timestamp: new Date().toISOString()
        };

        try {
            sessionStorage.setItem(key, JSON.stringify(fileData));
            return true;
        } catch (e) {
            console.error('Error saving file to storage:', e);
            // Handle quota exceeded error for large files
            if (e.name === 'QuotaExceededError') {
                alert('File is too large to store. Please use a smaller file.');
            }
            return false;
        }
    },

    /**
     * Get uploaded file data from sessionStorage
     * @param {string} type - Media type (image, video, audio, text, url)
     * @returns {Object|null} - File data object or null if not found
     */
    getFile: function (type) {
        const key = this.KEYS[type];
        if (!key) {
            console.error('Invalid file type:', type);
            return null;
        }

        try {
            const data = sessionStorage.getItem(key);
            return data ? JSON.parse(data) : null;
        } catch (e) {
            console.error('Error retrieving file from storage:', e);
            return null;
        }
    },

    /**
     * Clear stored file data
     * @param {string} type - Media type to clear (optional, clears all if not specified)
     */
    clearFile: function (type) {
        if (type) {
            const key = this.KEYS[type];
            if (key) {
                sessionStorage.removeItem(key);
            }
        } else {
            // Clear all VisioNova storage
            Object.values(this.KEYS).forEach(key => {
                sessionStorage.removeItem(key);
            });
        }
    },

    /**
     * Check if a file exists in storage
     * @param {string} type - Media type to check
     * @returns {boolean}
     */
    hasFile: function (type) {
        return this.getFile(type) !== null;
    },

    /**
     * Read file as base64 data URL
     * @param {File} file - File object from input
     * @returns {Promise<string>} - Base64 data URL
     */
    readFileAsDataURL: function (file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result);
            reader.onerror = () => reject(reader.error);
            reader.readAsDataURL(file);
        });
    }
};

// Make available globally
window.VisioNovaStorage = VisioNovaStorage;
