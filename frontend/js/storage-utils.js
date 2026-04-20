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
    },

    /**
     * Initialize IndexedDB for File storage
     */
    _initDB: function () {
        return new Promise((resolve, reject) => {
            const request = indexedDB.open('VisioNovaDB', 1);

            request.onerror = () => reject(request.error);
            request.onsuccess = () => resolve(request.result);

            request.onupgradeneeded = (event) => {
                const db = event.target.result;
                if (!db.objectStoreNames.contains('files')) {
                    db.createObjectStore('files', { keyPath: 'id' });
                }
            };
        });
    },

    /**
     * Store a File object in IndexedDB for document uploads (PDF, DOCX)
     * @param {File} file - The File object to store
     * @param {string} fileName - Original filename
     */
    saveDocumentFile: async function (file, fileName) {
        try {
            const db = await this._initDB();
            const transaction = db.transaction(['files'], 'readwrite');
            const store = transaction.objectStore('files');

            const fileData = {
                id: 'currentDocument',
                file: file,
                fileName: fileName || file.name,
                mimeType: file.type,
                size: file.size,
                timestamp: new Date().toISOString()
            };

            await new Promise((resolve, reject) => {
                const request = store.put(fileData);
                request.onsuccess = () => resolve();
                request.onerror = () => reject(request.error);
            });

            // Also store metadata in sessionStorage as backup
            sessionStorage.setItem('visioNova_documentFile_meta', JSON.stringify({
                fileName: fileData.fileName,
                mimeType: fileData.mimeType,
                size: file.size,
                timestamp: fileData.timestamp
            }));

            console.log('[Storage] Document file saved to IndexedDB:', fileName);
            db.close();
            return true;
        } catch (error) {
            console.error('[Storage] Error saving document file:', error);
            return false;
        }
    },

    getDocumentFile: async function () {
        try {
            const db = await this._initDB();
            const transaction = db.transaction(['files'], 'readonly');
            const store = transaction.objectStore('files');

            const fileData = await new Promise((resolve, reject) => {
                const request = store.get('currentDocument');
                request.onsuccess = () => resolve(request.result);
                request.onerror = () => reject(request.error);
            });

            db.close();
            console.log('[Storage] Document file retrieved from IndexedDB:', fileData ? fileData.fileName : 'none');
            return fileData || null;
        } catch (error) {
            console.error('[Storage] Error getting document file:', error);
            return null;
        }
    },

    hasDocumentFile: async function () {
        const fileData = await this.getDocumentFile();
        return fileData !== null;
    },

    clearDocumentFile: async function () {
        try {
            const db = await this._initDB();
            const transaction = db.transaction(['files'], 'readwrite');
            const store = transaction.objectStore('files');

            await new Promise((resolve, reject) => {
                const request = store.delete('currentDocument');
                request.onsuccess = () => resolve();
                request.onerror = () => reject(request.error);
            });

            sessionStorage.removeItem('visioNova_documentFile_meta');
            console.log('[Storage] Document file cleared from IndexedDB');
            db.close();
        } catch (error) {
            console.error('[Storage] Error clearing document file:', error);
        }
    },

    // ─── Image file storage (IndexedDB for large files) ──────────

    /**
     * Save image file data to IndexedDB (handles files up to 500 MB).
     * @param {string} dataURL - Base64 data URL of the image
     * @param {string} fileName - Original file name
     * @param {string} mimeType - MIME type
     * @returns {Promise<boolean>}
     */
    saveImageFile: async function (dataURL, fileName, mimeType) {
        try {
            const db = await this._initDB();
            const transaction = db.transaction(['files'], 'readwrite');
            const store = transaction.objectStore('files');

            const fileData = {
                id: 'currentImage',
                data: dataURL,
                fileName: fileName || 'image.png',
                mimeType: mimeType || 'image/png',
                timestamp: new Date().toISOString()
            };

            await new Promise((resolve, reject) => {
                const request = store.put(fileData);
                request.onsuccess = () => resolve();
                request.onerror = () => reject(request.error);
            });

            // Store lightweight metadata in sessionStorage for quick checks
            sessionStorage.setItem('visioNova_imageFile_meta', JSON.stringify({
                fileName: fileData.fileName,
                mimeType: fileData.mimeType,
                timestamp: fileData.timestamp
            }));

            console.log('[Storage] Image file saved to IndexedDB:', fileName);
            db.close();
            return true;
        } catch (error) {
            console.error('[Storage] Error saving image file to IndexedDB:', error);
            return false;
        }
    },

    /**
     * Retrieve image file data from IndexedDB.
     * @returns {Promise<Object|null>} - { data, fileName, mimeType, timestamp } or null
     */
    getImageFile: async function () {
        try {
            const db = await this._initDB();
            const transaction = db.transaction(['files'], 'readonly');
            const store = transaction.objectStore('files');

            const fileData = await new Promise((resolve, reject) => {
                const request = store.get('currentImage');
                request.onsuccess = () => resolve(request.result);
                request.onerror = () => reject(request.error);
            });

            db.close();
            if (fileData) {
                console.log('[Storage] Image file retrieved from IndexedDB:', fileData.fileName);
            }
            return fileData || null;
        } catch (error) {
            console.error('[Storage] Error getting image file from IndexedDB:', error);
            return null;
        }
    },

    /**
     * Check if an image file exists in IndexedDB
     * @returns {Promise<boolean>}
     */
    hasImageFile: async function () {
        const meta = sessionStorage.getItem('visioNova_imageFile_meta');
        if (meta) return true;
        const data = await this.getImageFile();
        return data !== null;
    },

    /**
     * Clear image file from IndexedDB
     */
    clearImageFile: async function () {
        try {
            const db = await this._initDB();
            const transaction = db.transaction(['files'], 'readwrite');
            const store = transaction.objectStore('files');

            await new Promise((resolve, reject) => {
                const request = store.delete('currentImage');
                request.onsuccess = () => resolve();
                request.onerror = () => reject(request.error);
            });

            sessionStorage.removeItem('visioNova_imageFile_meta');
            // Also clear the legacy sessionStorage key
            sessionStorage.removeItem(this.KEYS.image);
            console.log('[Storage] Image file cleared from IndexedDB');
            db.close();
        } catch (error) {
            console.error('[Storage] Error clearing image file:', error);
        }
    },

    // ─── Audio file storage (IndexedDB for large files) ───────────

    /**
     * Save audio file data to IndexedDB (handles files up to 200 MB).
     * Falls back to sessionStorage for tiny files.
     * @param {string} dataURL - Base64 data URL of the audio
     * @param {string} fileName - Original file name
     * @param {string} mimeType - MIME type
     * @returns {Promise<boolean>}
     */
    
    /**
     * Save a video file to IndexedDB
     */
    saveVideoFile: async function (dataURL, fileName, mimeType) {
        try {
            const db = await this._initDB();
            const transaction = db.transaction(['files'], 'readwrite');
            const store = transaction.objectStore('files');
            
            const fileData = {
                id: 'currentVideo',
                data: dataURL,
                fileName: fileName || 'video.mp4',
                mimeType: mimeType || 'video/mp4',
                timestamp: new Date().toISOString()
            };
            
            await new Promise((resolve, reject) => {
                const request = store.put(fileData);
                request.onsuccess = () => resolve();
                request.onerror = () => reject(request.error);
            });
            
            sessionStorage.setItem('visioNova_videoFile_meta', JSON.stringify({
                fileName: fileData.fileName,
                mimeType: fileData.mimeType,
                timestamp: fileData.timestamp
            }));
            
            console.log('[Storage] Video file saved to IndexedDB:', fileName);
            db.close();
            return true;
        } catch (error) {
            console.error('[Storage] Error saving video file:', error);
            return false;
        }
    },

    /**
     * Retrieve the current video file from IndexedDB
     */
    getVideoFile: async function () {
        try {
            const db = await this._initDB();
            const transaction = db.transaction(['files'], 'readonly');
            const store = transaction.objectStore('files');
            
            const fileData = await new Promise((resolve, reject) => {
                const request = store.get('currentVideo');
                request.onsuccess = () => resolve(request.result);
                request.onerror = () => reject(request.error);
            });
            
            db.close();
            return fileData;
        } catch (error) {
            console.error('[Storage] Error retrieving video file:', error);
            return null;
        }
    },

    saveAudioFile: async function (dataURL, fileName, mimeType) {
        try {
            const db = await this._initDB();
            const transaction = db.transaction(['files'], 'readwrite');
            const store = transaction.objectStore('files');

            const fileData = {
                id: 'currentAudio',
                data: dataURL,
                fileName: fileName || 'audio.wav',
                mimeType: mimeType || 'audio/wav',
                timestamp: new Date().toISOString()
            };

            await new Promise((resolve, reject) => {
                const request = store.put(fileData);
                request.onsuccess = () => resolve();
                request.onerror = () => reject(request.error);
            });

            // Store lightweight metadata in sessionStorage for quick checks
            sessionStorage.setItem('visioNova_audioFile_meta', JSON.stringify({
                fileName: fileData.fileName,
                mimeType: fileData.mimeType,
                timestamp: fileData.timestamp
            }));

            console.log('[Storage] Audio file saved to IndexedDB:', fileName);
            db.close();
            return true;
        } catch (error) {
            console.error('[Storage] Error saving audio file to IndexedDB:', error);
            return false;
        }
    },

    /**
     * Retrieve audio file data from IndexedDB.
     * @returns {Promise<Object|null>} - { data, fileName, mimeType, timestamp } or null
     */
    getAudioFile: async function () {
        try {
            const db = await this._initDB();
            const transaction = db.transaction(['files'], 'readonly');
            const store = transaction.objectStore('files');

            const fileData = await new Promise((resolve, reject) => {
                const request = store.get('currentAudio');
                request.onsuccess = () => resolve(request.result);
                request.onerror = () => reject(request.error);
            });

            db.close();
            if (fileData) {
                console.log('[Storage] Audio file retrieved from IndexedDB:', fileData.fileName);
            }
            return fileData || null;
        } catch (error) {
            console.error('[Storage] Error getting audio file from IndexedDB:', error);
            return null;
        }
    },

    /**
     * Check if an audio file exists in IndexedDB
     * @returns {Promise<boolean>}
     */
    hasAudioFile: async function () {
        const meta = sessionStorage.getItem('visioNova_audioFile_meta');
        if (meta) return true;
        const data = await this.getAudioFile();
        return data !== null;
    },

    /**
     * Clear audio file from IndexedDB
     */
    clearAudioFile: async function () {
        try {
            const db = await this._initDB();
            const transaction = db.transaction(['files'], 'readwrite');
            const store = transaction.objectStore('files');

            await new Promise((resolve, reject) => {
                const request = store.delete('currentAudio');
                request.onsuccess = () => resolve();
                request.onerror = () => reject(request.error);
            });

            sessionStorage.removeItem('visioNova_audioFile_meta');
            // Also clear the legacy sessionStorage key
            sessionStorage.removeItem(this.KEYS.audio);
            console.log('[Storage] Audio file cleared from IndexedDB');
            db.close();
        } catch (error) {
            console.error('[Storage] Error clearing audio file:', error);
        }
    },

    // ─── Result storage (IndexedDB for persistence) ──────────

    /**
     * Save analysis result to IndexedDB
     * @param {string} type - Media type (image, video, audio, text, url)
     * @param {Object} result - The analysis result object from the API
     */
    saveResult: async function (type, result) {
        // Fallback to SessionStorage for instant access across tabs
        try {
            sessionStorage.setItem(`visioNova_${type}_result`, JSON.stringify(result));
        } catch (e) {
            console.error("Fallback to sessionStorage failed:", e);
        }

        try {
            const db = await this._initDB();
            const transaction = db.transaction(['files'], 'readwrite');
            const store = transaction.objectStore('files');

            const resultData = {
                id: `result_${type}`,
                data: result,
                timestamp: new Date().toISOString()
            };

            await new Promise((resolve, reject) => {
                const request = store.put(resultData);
                request.onsuccess = () => resolve();
                request.onerror = () => reject(request.error);
            });

            console.log(`[Storage] Result for ${type} saved to IndexedDB`);
            db.close();
            return true;
        } catch (error) {
            console.error(`[Storage] Error saving result for ${type}:`, error);
            return false;
        }
    },

    /**
     * Get analysis result from IndexedDB (with fallback to sessionStorage)
     * @param {string} type - Media type (image, video, audio, text, url)
     * @returns {Promise<Object|null>} - The analysis result object
     */
    getResult: async function (type) {
        try {
            // First check IndexedDB
            const db = await this._initDB();
            const transaction = db.transaction(['files'], 'readonly');
            const store = transaction.objectStore('files');

            const resultData = await new Promise((resolve, reject) => {
                const request = store.get(`result_${type}`);
                request.onsuccess = () => resolve(request.result);
                request.onerror = () => reject(request.error);
            });

            db.close();
            
            if (resultData && resultData.data) {
                console.log(`[Storage] Result for ${type} retrieved from IndexedDB`);
                return resultData.data;
            }
        } catch (error) {
            console.error(`[Storage] Error getting result for ${type} from IndexedDB:`, error);
        }

        // Fallback to sessionStorage
        try {
            const sessionData = sessionStorage.getItem(`visioNova_${type}_result`);
            if (sessionData) {
                console.log(`[Storage] Result for ${type} retrieved from sessionStorage fallback`);
                return JSON.parse(sessionData);
            }
        } catch (error) {
            console.error(`[Storage] Error getting result for ${type} from sessionStorage:`, error);
        }
        
        return null;
    }
};

// Make available globally
window.VisioNovaStorage = VisioNovaStorage;
