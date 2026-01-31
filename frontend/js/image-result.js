/**
 * VisioNova Result Page - Image Analysis
 * Handles image analysis using the backend API and displays real results
 */

// API Configuration - Use Flask backend port
const API_BASE_URL = 'http://localhost:5000';

document.addEventListener('DOMContentLoaded', async function () {
    const imageData = VisioNovaStorage.getFile('image');

    if (imageData) {
        // Update page title with filename
        updateElement('pageTitle', 'Analyzing: ' + imageData.fileName);

        // Update timestamp
        const date = new Date(imageData.timestamp);
        updateElement('analysisTime', date.toLocaleDateString() + ' at ' + date.toLocaleTimeString());

        // Display the uploaded image
        const uploadedImage = document.getElementById('uploadedImage');
        const placeholder = document.getElementById('noImagePlaceholder');

        if (uploadedImage) {
            uploadedImage.src = imageData.data;
            uploadedImage.classList.remove('hidden');
        }
        if (placeholder) {
            placeholder.classList.add('hidden');
        }

        // Show loading state
        showLoadingState();

        // Call the backend API for analysis
        try {
            const analysisResult = await analyzeImage(imageData);
            displayAnalysisResults(analysisResult, imageData);
        } catch (error) {
            console.error('Analysis failed:', error);
            // Show real error message to user - do NOT fall back to mock data
            const errorMessage = error.message || 'Could not connect to backend. Ensure Flask server is running on port 5000.';
            displayError('API Error: ' + errorMessage);
            // Display failed state in UI
            displayFailedState(errorMessage);
        }

        // Load text detection result (if AnalysisDashboard stored it)
        loadTextDetectionResult();

        // Setup interactive image controls (Zoom, Pan, View Modes)
        setupImageInteractions();
    } else {
        updateElement('analysisTime', 'No analysis performed');
        updateElement('pageTitle', 'Image Analysis');
    }
});

/**
 * Call the backend API to analyze the image
 */
async function analyzeImage(imageData) {
    const response = await fetch(`${API_BASE_URL}/api/detect-image`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            image: imageData.data,
            filename: imageData.fileName,
            include_ela: true,
            include_metadata: true,
            include_watermark: true,
            include_c2pa: true
        })
    });

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `API error: ${response.status} ${response.statusText}`);
    }

    return await response.json();
}

/**
 * Show loading state while analyzing
 */
function showLoadingState() {
    const scoreElement = document.querySelector('.text-5xl.font-black, .text-4xl.font-black');
    if (scoreElement) {
        scoreElement.textContent = '...';
    }

    // Add loading spinner or pulse animation
    const verdictElements = document.querySelectorAll('[class*="text-accent-green"], [class*="text-success"]');
    verdictElements.forEach(el => {
        if (el.textContent.includes('Authentic') || el.textContent.includes('LIKELY')) {
            el.textContent = 'ANALYZING...';
            el.className = el.className.replace(/text-(accent-green|success|accent-red|danger)/g, 'text-gray-400');
        }
    });
}

/**
 * Display failed state when API call fails
 */
function displayFailedState(errorMsg) {
    // Update score to show failure
    updateElement('scoreValue', '?');

    // Update score circle to gray
    const scoreCircle = document.getElementById('scoreCircle');
    if (scoreCircle) {
        scoreCircle.setAttribute('stroke-dasharray', '0, 100');
        scoreCircle.classList.remove('text-accent-green', 'text-red-500');
        scoreCircle.classList.add('text-gray-500');
    }

    // Update glow to gray
    const scoreGlow = document.getElementById('scoreGlow');
    if (scoreGlow) {
        scoreGlow.className = 'absolute top-0 right-0 w-32 h-32 bg-gray-500/10 rounded-full blur-3xl -mr-10 -mt-10';
    }

    // Update verdict badge
    const verdictBadge = document.getElementById('verdictBadge');
    if (verdictBadge) {
        verdictBadge.textContent = 'ERROR';
        verdictBadge.className = 'inline-block px-3 py-1 rounded-full bg-yellow-500/20 text-yellow-400 text-xs font-bold border border-yellow-500/30';
    }

    // Update verdict description with ACTUAL error
    const friendlyError = errorMsg ? errorMsg : 'Could not connect to the analysis backend. Please ensure the Flask server is running on http://localhost:5000';
    updateElement('verdictDescription', friendlyError);

    // Update status badge
    const statusBadge = document.getElementById('statusBadge');
    if (statusBadge) {
        statusBadge.textContent = 'FAILED';
        statusBadge.className = 'px-2 py-0.5 rounded text-[10px] font-bold bg-red-500/10 text-red-400 border border-red-500/20';
    }

    // Update cards to show error state
    updateElement('aiProbValue', '--');
    updateElement('manipValue', '--');
    updateElement('manipDetail', 'Check backend');
    updateElement('metaValue', '--');
    updateElement('metaDetail', 'Check backend');
    updateElement('forensicsValue', '--');
    updateElement('forensicsDetail', 'Check backend');

    // Update watermark status
    const watermarkStatus = document.getElementById('watermarkStatus');
    if (watermarkStatus) {
        watermarkStatus.textContent = 'Error';
        watermarkStatus.className = 'text-[10px] px-2 py-0.5 rounded bg-red-500/20 text-red-400';
    }

    // Update C2PA status
    const c2paStatus = document.getElementById('c2paStatus');
    if (c2paStatus) {
        c2paStatus.textContent = 'Error';
        c2paStatus.className = 'text-[10px] px-2 py-0.5 rounded bg-red-500/20 text-red-400';
    }
}

/**
 * Display real analysis results from the API
 */
function displayAnalysisResults(result, fileData) {
    // Debug: Log full result to console
    console.log('Analysis Result:', result);

    if (!result.success) {
        displayError(result.error || 'Analysis failed');
        return;
    }

    const aiProbability = result.ai_probability || 50;
    const authenticityScore = Math.round(100 - aiProbability);
    const isLikelyAI = aiProbability > 50;
    const verdict = result.verdict || (isLikelyAI ? 'LIKELY_AI' : 'LIKELY_REAL');

    // Update status badge to COMPLETED
    const statusBadge = document.getElementById('statusBadge');
    if (statusBadge) {
        statusBadge.textContent = 'COMPLETED';
        statusBadge.className = 'px-2 py-0.5 rounded text-[10px] font-bold bg-accent-green/10 text-accent-green border border-accent-green/20';
    }

    // Generate and show analysis ID
    const analysisId = document.getElementById('analysisId');
    if (analysisId) {
        const id = Math.random().toString(36).substring(2, 10).toUpperCase();
        analysisId.textContent = `ID: #${id}`;
    }

    // Show anomaly marker if AI detected or manipulation found
    const anomalyMarker = document.getElementById('anomalyMarker');
    if (anomalyMarker && isLikelyAI) {
        anomalyMarker.classList.remove('hidden');
    }

    // Update page title
    updateElement('pageTitle', 'Analysis: ' + fileData.fileName);

    // Update overall score value
    const scoreValue = document.getElementById('scoreValue');
    if (scoreValue) {
        scoreValue.textContent = authenticityScore;
    }

    // Update score circle (SVG path)
    const scoreCircle = document.getElementById('scoreCircle');
    if (scoreCircle) {
        scoreCircle.setAttribute('stroke-dasharray', `${authenticityScore}, 100`);
        // Change color based on score
        if (isLikelyAI) {
            scoreCircle.classList.remove('text-accent-green');
            scoreCircle.classList.add('text-red-500');
            scoreCircle.className = scoreCircle.className.replace('drop-shadow-[0_0_8px_rgba(0,217,145,0.5)]', 'drop-shadow-[0_0_8px_rgba(239,68,68,0.5)]');
        } else {
            scoreCircle.classList.remove('text-red-500');
            scoreCircle.classList.add('text-accent-green');
        }
    }

    // Update glow color
    const scoreGlow = document.getElementById('scoreGlow');
    if (scoreGlow) {
        scoreGlow.className = isLikelyAI ?
            'absolute top-0 right-0 w-32 h-32 bg-red-500/10 rounded-full blur-3xl -mr-10 -mt-10' :
            'absolute top-0 right-0 w-32 h-32 bg-accent-green/10 rounded-full blur-3xl -mr-10 -mt-10';
    }

    // Update verdict badge
    const verdictBadge = document.getElementById('verdictBadge');
    if (verdictBadge) {
        const verdictText = getVerdictText(verdict);
        verdictBadge.textContent = verdictText;
        if (isLikelyAI) {
            verdictBadge.className = 'inline-block px-3 py-1 rounded-full bg-red-500/20 text-red-400 text-xs font-bold border border-red-500/30';
        } else {
            verdictBadge.className = 'inline-block px-3 py-1 rounded-full bg-accent-green/20 text-accent-green text-xs font-bold border border-accent-green/30';
        }
    }

    // Update verdict description
    const verdictDesc = document.getElementById('verdictDescription');
    if (verdictDesc) {
        verdictDesc.textContent = result.verdict_description || getVerdictDescription(result, authenticityScore);
    }

    // Update AI Probability card
    updateElement('aiProbValue', `${Math.round(aiProbability)}%`);
    const aiProbBar = document.getElementById('aiProbBar');
    if (aiProbBar) {
        aiProbBar.style.width = `${aiProbability}%`;
        aiProbBar.className = aiProbability > 50 ?
            'bg-red-500 h-full rounded-full transition-all duration-500' :
            'bg-accent-blue h-full rounded-full transition-all duration-500';
    }
    const aiProbIcon = document.getElementById('aiProbIcon');
    if (aiProbIcon) {
        aiProbIcon.className = aiProbability > 50 ?
            'material-symbols-outlined text-red-500 text-lg group-hover:scale-110 transition-transform' :
            'material-symbols-outlined text-accent-blue text-lg group-hover:scale-110 transition-transform';
    }

    // Update Manipulation card
    updateManipulationCard(result);

    // Update Metadata card
    updateMetadataCard(result);

    // Update Forensics card
    updateForensicsCard(result);

    // Update file info bar
    updateFileInfoBar(result, fileData);

    // Update analysis scores from API
    if (result.analysis_scores) {
        updateAnalysisScores(result.analysis_scores);
    }

    // Update metadata analysis
    if (result.metadata) {
        updateMetadataDisplay(result.metadata);
    }

    // Update ELA display
    if (result.ela) {
        updateELADisplay(result.ela);
    }

    // Update watermark detection display
    if (result.watermark) {
        updateWatermarkDisplay(result.watermark);
    } else {
        // Set default "not checked" state
        updateWatermarkDisplay({ watermark_detected: false, status: 'NOT_CHECKED' });
    }

    // Update Content Credentials (C2PA) display
    if (result.content_credentials) {
        updateC2PADisplay(result.content_credentials);
    } else {
        // Set default "not checked" state
        updateC2PADisplay({ c2pa_found: false, status: 'NOT_CHECKED' });
    }

    // Animate progress bars based on actual scores
    animateProgressBarsWithScores(result.analysis_scores || {});

    // Store result for potential export
    sessionStorage.setItem('visioNova_image_result', JSON.stringify(result));
}

/**
 * Get human-readable verdict text
 */
function getVerdictText(verdict) {
    const verdictMap = {
        'AI_GENERATED': 'AI GENERATED',
        'LIKELY_AI': 'LIKELY AI GENERATED',
        'UNCERTAIN': 'UNCERTAIN',
        'LIKELY_REAL': 'LIKELY AUTHENTIC',
        'REAL': 'AUTHENTIC',
        'ERROR': 'ANALYSIS ERROR'
    };
    return verdictMap[verdict] || verdict;
}

/**
 * Generate verdict description based on analysis results
 */
function getVerdictDescription(result, score) {
    const aiProb = result.ai_probability || 50;

    if (aiProb > 80) {
        return 'This image shows strong indicators of AI generation including synthetic patterns and potential watermarks.';
    } else if (aiProb > 50) {
        return 'This image shows some characteristics consistent with AI generation. Further manual review recommended.';
    } else if (aiProb > 30) {
        return 'This image shows minor inconsistencies but maintains reasonable integrity in metadata and pixel structure.';
    } else {
        return 'This image appears authentic with consistent metadata, natural noise patterns, and no AI generation signatures detected.';
    }
}

/**
 * Update the Manipulation card with real data
 */
function updateManipulationCard(result) {
    const manipValue = document.getElementById('manipValue');
    const manipDetail = document.getElementById('manipDetail');
    const manipIcon = document.getElementById('manipIcon');

    // Check for manipulation from ELA or metadata
    const ela = result.ela || {};
    const manipLevel = ela.manipulation_likelihood || 0;
    const cloneDetected = ela.clone_detected || false;

    if (cloneDetected || manipLevel > 60) {
        if (manipValue) manipValue.textContent = 'Detected';
        if (manipDetail) {
            manipDetail.textContent = cloneDetected ? 'Cloning artifacts found' : `${Math.round(manipLevel)}% likelihood`;
            manipDetail.className = 'text-[10px] text-red-400 mt-1';
        }
        if (manipIcon) {
            manipIcon.textContent = 'warning';
            manipIcon.className = 'material-symbols-outlined text-red-500 text-lg group-hover:scale-110 transition-transform';
        }
    } else if (manipLevel > 30) {
        if (manipValue) manipValue.textContent = 'Possible';
        if (manipDetail) {
            manipDetail.textContent = `${Math.round(manipLevel)}% likelihood`;
            manipDetail.className = 'text-[10px] text-yellow-400 mt-1';
        }
        if (manipIcon) {
            manipIcon.textContent = 'help';
            manipIcon.className = 'material-symbols-outlined text-yellow-400 text-lg group-hover:scale-110 transition-transform';
        }
    } else {
        if (manipValue) manipValue.textContent = 'None';
        if (manipDetail) {
            manipDetail.textContent = 'No cloning detected';
            manipDetail.className = 'text-[10px] text-[#8da8ce] mt-1';
        }
        if (manipIcon) {
            manipIcon.textContent = 'check_circle';
            manipIcon.className = 'material-symbols-outlined text-accent-green text-lg group-hover:scale-110 transition-transform';
        }
    }
}

/**
 * Update the Metadata card with real data
 */
function updateMetadataCard(result) {
    const metaValue = document.getElementById('metaValue');
    const metaDetail = document.getElementById('metaDetail');
    const metaIcon = document.getElementById('metaIcon');

    const metadata = result.metadata || {};
    const hasExif = metadata.has_exif || false;
    const isComplete = metadata.is_complete || false;
    const stripped = metadata.exif_stripped || false;
    const aiTool = metadata.ai_tool_detected;

    if (aiTool) {
        if (metaValue) metaValue.textContent = 'AI Tool';
        if (metaDetail) {
            metaDetail.textContent = `${aiTool} detected`;
            metaDetail.className = 'text-[10px] text-red-400 mt-1';
        }
        if (metaIcon) {
            metaIcon.textContent = 'warning';
            metaIcon.className = 'material-symbols-outlined text-red-500 text-lg group-hover:scale-110 transition-transform';
        }
    } else if (!hasExif || stripped) {
        if (metaValue) metaValue.textContent = 'Partial';
        if (metaDetail) {
            metaDetail.textContent = 'EXIF data stripped';
            metaDetail.className = 'text-[10px] text-yellow-400 mt-1';
        }
        if (metaIcon) {
            metaIcon.textContent = 'warning';
            metaIcon.className = 'material-symbols-outlined text-yellow-400 text-lg group-hover:scale-110 transition-transform';
        }
    } else if (isComplete) {
        if (metaValue) metaValue.textContent = 'Complete';
        if (metaDetail) {
            metaDetail.textContent = 'All metadata intact';
            metaDetail.className = 'text-[10px] text-[#8da8ce] mt-1';
        }
        if (metaIcon) {
            metaIcon.textContent = 'check_circle';
            metaIcon.className = 'material-symbols-outlined text-accent-green text-lg group-hover:scale-110 transition-transform';
        }
    } else {
        if (metaValue) metaValue.textContent = 'Present';
        if (metaDetail) {
            metaDetail.textContent = 'Basic metadata found';
            metaDetail.className = 'text-[10px] text-[#8da8ce] mt-1';
        }
        if (metaIcon) {
            metaIcon.textContent = 'info';
            metaIcon.className = 'material-symbols-outlined text-accent-blue text-lg group-hover:scale-110 transition-transform';
        }
    }
}

/**
 * Update the Forensics card with real data
 */
function updateForensicsCard(result) {
    const forensicsValue = document.getElementById('forensicsValue');
    const forensicsDetail = document.getElementById('forensicsDetail');
    const forensicsIcon = document.getElementById('forensicsIcon');

    const scores = result.analysis_scores || {};
    const noiseConsistency = scores.noise_consistency || 50;
    const ela = result.ela || {};
    const elaScore = ela.ela_score || 50;

    // Average forensics score
    const forensicsScore = (noiseConsistency + (100 - elaScore)) / 2;

    if (forensicsScore > 70) {
        if (forensicsValue) forensicsValue.textContent = 'Pass';
        if (forensicsDetail) {
            forensicsDetail.textContent = 'Noise consistency OK';
            forensicsDetail.className = 'text-[10px] text-[#8da8ce] mt-1';
        }
        if (forensicsIcon) {
            forensicsIcon.textContent = 'security';
            forensicsIcon.className = 'material-symbols-outlined text-accent-green text-lg group-hover:scale-110 transition-transform';
        }
    } else if (forensicsScore > 40) {
        if (forensicsValue) forensicsValue.textContent = 'Warning';
        if (forensicsDetail) {
            forensicsDetail.textContent = 'Minor anomalies found';
            forensicsDetail.className = 'text-[10px] text-yellow-400 mt-1';
        }
        if (forensicsIcon) {
            forensicsIcon.textContent = 'shield';
            forensicsIcon.className = 'material-symbols-outlined text-yellow-400 text-lg group-hover:scale-110 transition-transform';
        }
    } else {
        if (forensicsValue) forensicsValue.textContent = 'Fail';
        if (forensicsDetail) {
            forensicsDetail.textContent = 'Significant anomalies';
            forensicsDetail.className = 'text-[10px] text-red-400 mt-1';
        }
        if (forensicsIcon) {
            forensicsIcon.textContent = 'gpp_bad';
            forensicsIcon.className = 'material-symbols-outlined text-red-500 text-lg group-hover:scale-110 transition-transform';
        }
    }
}

/**
 * Update the file info bar at bottom of image
 */
function updateFileInfoBar(result, fileData) {
    const metadata = result.metadata || {};

    // Update resolution
    const resolution = document.getElementById('imgResolution');
    if (resolution) {
        if (metadata.width && metadata.height) {
            resolution.textContent = `${metadata.width} x ${metadata.height}px`;
        } else {
            resolution.textContent = 'Unknown';
        }
    }

    // Update file size
    const sizeEl = document.getElementById('imgSize');
    if (sizeEl) {
        sizeEl.textContent = formatFileSize(fileData.data.length * 0.75); // Approximate base64 to actual size
    }

    // Update file type
    const typeEl = document.getElementById('imgType');
    if (typeEl) {
        const ext = fileData.fileName.split('.').pop()?.toUpperCase() || 'IMG';
        typeEl.textContent = result.format?.toUpperCase() || ext;
    }
}

/**
 * Update analysis score displays
 */
function updateAnalysisScores(scores) {
    // Map score names to display elements
    const scoreMapping = {
        'noise_consistency': 'noiseScore',
        'frequency_anomaly': 'frequencyScore',
        'color_uniformity': 'colorScore',
        'edge_naturalness': 'edgeScore',
        'texture_quality': 'textureScore'
    };

    for (const [apiKey, elementId] of Object.entries(scoreMapping)) {
        if (scores[apiKey] !== undefined) {
            const value = Math.round(scores[apiKey]);
            updateElement(elementId, `${value}%`);

            // Update progress bar if exists
            const bar = document.getElementById(`${elementId}Bar`);
            if (bar) {
                bar.style.width = `${value}%`;
                bar.style.backgroundColor = value > 60 ? '#FF4A4A' : value > 30 ? '#FFC107' : '#00D991';
            }
        }
    }
}

/**
 * Update metadata display section
 */
function updateMetadataDisplay(metadata) {
    const metadataSection = document.getElementById('metadataSection');
    if (!metadataSection) return;

    // Update EXIF status
    const exifStatus = document.getElementById('exifStatus');
    if (exifStatus) {
        if (metadata.has_exif) {
            exifStatus.textContent = 'EXIF Data Present';
            exifStatus.className = exifStatus.className.replace(/text-\w+-\d+/g, 'text-green-400');
        } else {
            exifStatus.textContent = 'No EXIF Data (Suspicious)';
            exifStatus.className = exifStatus.className.replace(/text-\w+-\d+/g, 'text-red-400');
        }
    }

    // Update camera info
    if (metadata.camera_make || metadata.camera_model) {
        updateElement('cameraInfo', `${metadata.camera_make || ''} ${metadata.camera_model || ''}`.trim());
    } else {
        updateElement('cameraInfo', 'No camera information');
    }

    // Update software detection
    if (metadata.ai_software_detected) {
        updateElement('softwareInfo', `AI Software Detected: ${metadata.software_detected}`);
        const softwareEl = document.getElementById('softwareInfo');
        if (softwareEl) softwareEl.className = softwareEl.className.replace(/text-\w+-\d+/g, 'text-red-400');
    } else if (metadata.software_detected) {
        updateElement('softwareInfo', `Software: ${metadata.software_detected}`);
    }

    // NEW: Display Risk Adjustment if significant
    const riskDiff = metadata.ai_probability_modifier || 0;
    if (Math.abs(riskDiff) > 5) {
        const riskEl = document.createElement('div');
        riskEl.className = 'flex justify-between items-center bg-[#0f121a]/50 p-2 rounded-lg border border-[#20324b] mt-2';
        const isRisk = riskDiff > 0;
        riskEl.innerHTML = `
            <span class="text-[10px] text-[#8da8ce] uppercase">Risk Adjustment</span>
            <span class="text-xs font-mono ${isRisk ? 'text-red-400' : 'text-green-400'}">
                ${isRisk ? '+' : ''}${riskDiff}% ${isRisk ? '(AI Traits)' : '(Authentic Traits)'}
            </span>
        `;
        // Insert after camera/software info (which is in a grid)
        const container = document.getElementById('metadataGrid');
        if (container) container.parentNode.insertBefore(riskEl, container.nextSibling);
    }

    // Display anomalies
    if (metadata.anomalies && metadata.anomalies.length > 0) {
        const anomaliesList = document.getElementById('anomaliesList');
        if (anomaliesList) {
            anomaliesList.innerHTML = metadata.anomalies
                .map(a => `<li class="text-yellow-400 text-sm">⚠️ ${a}</li>`)
                .join('');
        }
    }
}

/**
 * Setup interactive image controls (Zoom, Pan, View Modes)
 */
function setupImageInteractions() {
    // 1. View Mode Switching
    const modes = {
        'original': { btn: 'btnViewOriginal', img: 'uploadedImage' },
        'heatmap': { btn: 'btnViewHeatmap', img: 'heatmapImage' },
        'ela': { btn: 'btnViewELA', img: 'elaImage' }
    };

    let currentMode = 'original';

    // Helper to switch mode
    const switchMode = async (mode) => {
        // Update buttons
        Object.entries(modes).forEach(([m, data]) => {
            const btn = document.getElementById(data.btn);
            if (!btn) return;

            if (m === mode) {
                btn.classList.remove('bg-[#20324b]', 'text-[#8da8ce]');
                btn.classList.add('bg-primary', 'text-white', 'shadow-md', 'shadow-primary/20');
            } else {
                btn.classList.add('bg-[#20324b]', 'text-[#8da8ce]');
                btn.classList.remove('bg-primary', 'text-white', 'shadow-md', 'shadow-primary/20');
            }
        });

        // specific logic for heatmap lazy loading
        if (mode === 'heatmap') {
            const heatmapImg = document.getElementById('heatmapImage');
            if (heatmapImg && !heatmapImg.src) {
                // Fetch heatmap if missing
                try {
                    const btn = document.getElementById('btnViewHeatmap');
                    const originalText = btn.innerHTML;
                    btn.innerHTML = '<span class="material-symbols-outlined text-[18px] animate-spin">refresh</span> Loading...';

                    const imageData = VisioNovaStorage.getFile('image');
                    if (imageData) {
                        const response = await fetch(`${API_BASE_URL}/api/detect-image/ela`, {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ image: imageData.data, colormap: 'jet' })
                        });
                        const data = await response.json();
                        if (data.success && data.ela_heatmap) {
                            heatmapImg.src = `data:image/png;base64,${data.ela_heatmap}`;
                        }
                    }
                    btn.innerHTML = originalText;
                } catch (e) {
                    console.error('Failed to load heatmap', e);
                }
            }
        }

        // Update Images Visibility
        Object.entries(modes).forEach(([m, data]) => {
            const img = document.getElementById(data.img);
            if (img) {
                if (m === mode) {
                    img.classList.remove('hidden');
                    // Small timeout to allow transition
                    setTimeout(() => img.classList.remove('opacity-0'), 10);
                } else {
                    img.classList.add('opacity-0');
                    setTimeout(() => img.classList.add('hidden'), 300);
                }
            }
        });

        currentMode = mode;
    };

    // Attach listeners
    Object.keys(modes).forEach(mode => {
        const btn = document.getElementById(modes[mode].btn);
        if (btn) btn.addEventListener('click', () => switchMode(mode));
    });

    // 2. Zoom & Pan Controls
    let scale = 1;
    const ZOOM_STEP = 0.2;
    const wrapper = document.getElementById('imageWrapper');

    if (wrapper) {
        document.getElementById('btnZoomIn')?.addEventListener('click', () => {
            scale = Math.min(scale + ZOOM_STEP, 5);
            updateTransform();
        });

        document.getElementById('btnZoomOut')?.addEventListener('click', () => {
            scale = Math.max(scale - ZOOM_STEP, 0.5);
            updateTransform();
        });

        document.getElementById('btnResetView')?.addEventListener('click', () => {
            scale = 1;
            updateTransform();
        });

        document.getElementById('btnFullscreen')?.addEventListener('click', () => {
            const container = document.getElementById('mainImageContainer');
            if (document.fullscreenElement) {
                document.exitFullscreen();
            } else if (container) {
                container.requestFullscreen();
            }
        });

        function updateTransform() {
            wrapper.style.transform = `scale(${scale})`;
        }
    }
}

/**
 * Update ELA display section
 */
function updateELADisplay(ela) {
    if (!ela.success) return;

    // Update ELA image if element exists (Raw ELA)
    const elaImage = document.getElementById('elaImage');
    if (elaImage && ela.ela_image) {
        elaImage.src = `data:image/png;base64,${ela.ela_image}`;
        // Do NOT remove hidden here, let the view toggler handle it
    }

    // Update manipulation likelihood
    updateElement('manipulationScore', `${Math.round(ela.manipulation_likelihood || 0)}%`);

    // Update suspicious regions count
    if (ela.suspicious_regions) {
        updateElement('suspiciousRegions', `${ela.suspicious_regions.length} region(s) detected`);
    }
}

/**
 * Update Watermark detection display
 */
/**
 * Update Watermark detection display
 */
function updateWatermarkDisplay(watermark) {
    const statusEl = document.getElementById('watermarkStatus');
    const detectedEl = document.getElementById('wm_detected');
    const typeEl = document.getElementById('wm_type');
    const confidenceEl = document.getElementById('wm_confidence');
    const detailsList = document.getElementById('wm_details_list');

    // Clear previous details
    if (detailsList) detailsList.innerHTML = '';

    if (watermark.watermark_detected) {
        if (statusEl) {
            statusEl.textContent = 'FOUND';
            statusEl.className = 'text-[10px] px-2 py-0.5 rounded bg-red-500/20 text-red-400 font-medium';
        }
        if (detectedEl) {
            detectedEl.textContent = 'Yes';
            detectedEl.className = 'text-red-400 font-semibold';
        }
        if (typeEl) {
            typeEl.textContent = watermark.watermark_type || 'Unknown';
            typeEl.className = 'text-white';
        }
        if (confidenceEl) {
            confidenceEl.textContent = `${watermark.confidence || 0}%`;
            confidenceEl.className = watermark.confidence > 70 ? 'text-red-400 font-semibold' : 'text-yellow-400';
        }

        // Add AI generator signature if found
        if (watermark.ai_generator_signature && detailsList) {
            detailsList.innerHTML += `<div class="flex items-center gap-2 text-red-400 mt-2 p-2 bg-red-500/10 rounded-lg border border-red-500/20">
                <span class="material-symbols-outlined text-sm">warning</span>
                <span>AI Generator: <strong>${watermark.ai_generator_signature}</strong></span>
            </div>`;
        }
    } else {
        if (statusEl) {
            statusEl.textContent = 'NOT FOUND';
            statusEl.className = 'text-[10px] px-2 py-0.5 rounded bg-green-500/20 text-green-400 font-medium';
        }
        if (detectedEl) {
            detectedEl.textContent = 'No';
            detectedEl.className = 'text-green-400';
        }
        if (typeEl) {
            typeEl.textContent = 'None detected';
            typeEl.className = 'text-[#8da8ce]';
        }
        if (confidenceEl) {
            confidenceEl.textContent = 'N/A';
            confidenceEl.className = 'text-[#8da8ce]';
        }
    }

    // Add detection method details
    if (detailsList && watermark.details && watermark.details.length > 0) {
        const detailsHtml = watermark.details.map(d => {
            const isWarning = d.includes('⚠️') || d.includes('Experimental');
            const isMatch = d.includes('Matched') || d.includes('detected');
            const colorClass = isWarning ? 'text-yellow-400' : (isMatch ? 'text-red-400' : 'text-[#8da8ce]');
            return `<div class="${colorClass} text-xs">• ${d}</div>`;
        }).join('');
        detailsList.innerHTML += `<div class="mt-2 space-y-1">${detailsHtml}</div>`;
    }

    // Show detection methods summary
    if (detailsList && watermark.detection_methods) {
        const methods = watermark.detection_methods;
        let methodsSummary = '<div class="mt-3 pt-2 border-t border-[#20324b]">';
        methodsSummary += '<p class="text-[#64748b] text-[10px] uppercase tracking-wider mb-2">Detection Methods:</p>';
        methodsSummary += '<div class="grid grid-cols-2 gap-2 text-[10px]">';

        const methodNames = {
            'invisible_watermark': 'DWT-DCT',
            'spectral_analysis': 'Spectral',
            'lsb_analysis': 'LSB Stats',
            'metadata_watermark': 'Metadata',
            'treering_analysis': 'Tree-Ring',
            'adversarial_analysis': 'Adversarial',
            'steganogan': 'SteganoGAN',
            'synthid': 'SynthID'
        };

        for (const [key, value] of Object.entries(methods)) {
            const name = methodNames[key] || key;
            const detected = value.detected || value.found || value.patterns_found || value.anomaly_detected;
            const icon = detected ? 'check_circle' : 'cancel';
            const color = detected ? 'text-red-400' : 'text-[#64748b]';
            const note = value.note ? `<span class="block text-[9px] text-yellow-500/70 ml-5">${value.note}</span>` : '';

            methodsSummary += `
            <div class="${detected ? 'bg-red-500/5 border-red-500/20' : 'bg-[#20324b]/30'} p-1.5 rounded border border-transparent">
                <div class="${color} flex items-center gap-1.5 font-medium">
                    <span class="material-symbols-outlined text-[14px]">${icon}</span>
                    ${name}
                </div>
                ${note}
            </div>`;
        }

        methodsSummary += '</div></div>';
        detailsList.innerHTML += methodsSummary;
    }
}

/**
 * Update Content Credentials (C2PA) display
 */
function updateC2PADisplay(c2pa) {
    const statusEl = document.getElementById('c2paStatus');
    const foundEl = document.getElementById('c2pa_found');
    const aiGenEl = document.getElementById('c2pa_ai_generated');
    const generatorEl = document.getElementById('c2pa_generator');
    const signatureEl = document.getElementById('c2pa_signature');
    const provenanceEl = document.getElementById('c2pa_provenance');

    // Clear previous content
    if (provenanceEl) provenanceEl.innerHTML = '';

    if (c2pa.c2pa_found || c2pa.has_content_credentials) {
        if (statusEl) {
            statusEl.textContent = 'VERIFIED';
            statusEl.className = 'text-[10px] px-2 py-0.5 rounded bg-blue-500/20 text-blue-400 font-medium';
        }
        if (foundEl) {
            foundEl.textContent = 'Yes';
            foundEl.className = 'text-blue-400 font-semibold';
        }

        if (c2pa.is_ai_generated) {
            if (aiGenEl) {
                aiGenEl.textContent = 'Yes';
                aiGenEl.className = 'text-red-400 font-semibold';
            }
            if (generatorEl) {
                generatorEl.textContent = c2pa.ai_generator || 'Unknown AI Tool';
                generatorEl.className = 'text-red-400 font-semibold';
            }
        } else {
            if (aiGenEl) {
                aiGenEl.textContent = 'No';
                aiGenEl.className = 'text-green-400';
            }
            if (generatorEl) {
                generatorEl.textContent = c2pa.generator_info?.name || 'Camera/Editor';
                generatorEl.className = 'text-[#8da8ce]';
            }
        }

        if (signatureEl) {
            if (c2pa.signature_valid === true) {
                signatureEl.textContent = 'Valid ✓';
                signatureEl.className = 'text-green-400 font-semibold';
            } else if (c2pa.signature_valid === false) {
                signatureEl.textContent = 'Invalid ✗';
                signatureEl.className = 'text-red-400 font-semibold';
            } else {
                signatureEl.textContent = 'Unknown';
                signatureEl.className = 'text-yellow-400';
            }
        }

        // Show provenance chain
        if (provenanceEl && c2pa.provenance_chain && c2pa.provenance_chain.length > 0) {
            provenanceEl.innerHTML = `
                <div class="mt-3 p-2 bg-blue-500/10 rounded-lg border border-blue-500/20">
                    <p class="text-blue-400 text-xs font-medium mb-2 flex items-center gap-1">
                        <span class="material-symbols-outlined text-sm">account_tree</span>
                        Provenance Chain
                    </p>
                    ${c2pa.provenance_chain.map((p, i) => `<div class="text-white text-xs pl-2 border-l-2 border-blue-500/30 mb-1">${i + 1}. ${p}</div>`).join('')}
                </div>
            `;
        }

        // Show trust indicators
        if (c2pa.trust_indicators && provenanceEl) {
            const trustScore = c2pa.trust_indicators.trust_score || 0;
            const trustColor = trustScore > 70 ? 'green' : (trustScore > 40 ? 'yellow' : 'red');
            provenanceEl.innerHTML += `
                <div class="mt-2 p-2 bg-[#20324b]/50 rounded-lg">
                    <div class="flex items-center justify-between mb-1">
                        <span class="text-[#8da8ce] text-xs">Trust Score</span>
                        <span class="text-${trustColor}-400 text-xs font-semibold">${trustScore}/100</span>
                    </div>
                    <div class="bg-[#20324b] h-1.5 rounded-full overflow-hidden">
                        <div class="bg-${trustColor}-400 h-full rounded-full transition-all duration-500" style="width: ${trustScore}%"></div>
                    </div>
                </div>
            `;
        }
    } else {
        if (statusEl) {
            statusEl.textContent = 'NOT FOUND';
            statusEl.className = 'text-[10px] px-2 py-0.5 rounded bg-[#20324b] text-[#8da8ce]';
        }
        if (foundEl) {
            foundEl.textContent = 'No';
            foundEl.className = 'text-[#8da8ce]';
        }
        if (aiGenEl) {
            aiGenEl.textContent = 'Unknown';
            aiGenEl.className = 'text-[#8da8ce]';
        }
        if (generatorEl) {
            generatorEl.textContent = 'N/A';
            generatorEl.className = 'text-[#8da8ce]';
        }
        if (signatureEl) {
            signatureEl.textContent = 'N/A';
            signatureEl.className = 'text-[#8da8ce]';
        }

        if (provenanceEl) {
            provenanceEl.innerHTML = `
                <div class="mt-2 p-2 bg-[#20324b]/30 rounded-lg text-[#64748b] text-xs">
                    <span class="material-symbols-outlined text-sm align-middle mr-1">info</span>
                    No C2PA Content Credentials found. This doesn't mean the image is fake - most images don't have C2PA yet.
                </div>
            `;
        }
    }

    // Show details
    if (provenanceEl && c2pa.details && c2pa.details.length > 0) {
        const detailsHtml = c2pa.details.map(d => `<div class="text-[#8da8ce] text-xs">• ${d}</div>`).join('');
        provenanceEl.innerHTML += `<div class="mt-2 space-y-1">${detailsHtml}</div>`;
    }
}

/**
 * Animate progress bars with actual API scores
 */
function animateProgressBarsWithScores(scores) {
    const bars = document.querySelectorAll('.bg-accent-blue, .bg-primary, [data-score-bar]');

    const scoreValues = Object.values(scores);
    bars.forEach((bar, index) => {
        const width = scoreValues[index % scoreValues.length] || 50;
        bar.style.transition = 'width 1s ease-out';
        bar.style.width = width + '%';

        // Color based on score (higher = more AI-like = red)
        if (width > 60) {
            bar.style.backgroundColor = '#FF4A4A';
        } else if (width > 30) {
            bar.style.backgroundColor = '#FFC107';
        } else {
            bar.style.backgroundColor = '#00D991';
        }
    });
}

/**
 * Display error message
 */
function displayError(message) {
    console.error('Analysis error:', message);

    const verdictElements = document.querySelectorAll('[class*="text-accent-green"], [class*="text-success"]');
    verdictElements.forEach(el => {
        if (el.textContent.includes('Authentic') || el.textContent.includes('LIKELY') || el.textContent.includes('ANALYZING')) {
            el.textContent = 'ANALYSIS FAILED';
            el.className = el.className.replace(/text-(accent-green|success|gray-400)/g, 'text-yellow-500');
        }
    });
}

/**
 * Fall back to mock results if API fails
 */
function displayMockResults(fileData) {
    const hash = hashString(fileData.fileName);
    const authenticityScore = 40 + (hash % 55);
    const fakeScore = 100 - authenticityScore;
    const isFake = fakeScore > 50;

    const scoreElement = document.querySelector('.text-5xl.font-black, .text-4xl.font-black');
    if (scoreElement) {
        scoreElement.textContent = authenticityScore;
    }

    const scoreCircle = document.querySelector('circle[stroke="#00D991"], circle[stroke="#FF4A4A"]');
    if (scoreCircle) {
        const circumference = 251.2;
        const offset = circumference - (authenticityScore / 100) * circumference;
        scoreCircle.setAttribute('stroke-dashoffset', offset);
        scoreCircle.setAttribute('stroke', isFake ? '#FF4A4A' : '#00D991');
    }

    animateProgressBars(hash);
}

/**
 * Animate progress bars with random values (fallback)
 */
function animateProgressBars(hash) {
    const bars = document.querySelectorAll('.bg-accent-blue, .bg-primary');
    bars.forEach((bar, index) => {
        const width = 30 + ((hash + index * 17) % 60);
        bar.style.transition = 'width 1s ease-out';
        bar.style.width = width + '%';
    });
}

/**
 * Helper to update element text content
 */
function updateElement(id, text) {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
}

/**
 * Simple string hash function
 */
function hashString(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
        const char = str.charCodeAt(i);
        hash = ((hash << 5) - hash) + char;
        hash = hash & hash;
    }
    return Math.abs(hash);
}

/**
 * Format file size
 */
function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / 1048576).toFixed(1) + ' MB';
}

/**
 * Load text detection result stored by AnalysisDashboard and display it
 */
function loadTextDetectionResult() {
    const raw = sessionStorage.getItem('visioNova_text_result');

    const predEl = document.getElementById('td_prediction');
    const confEl = document.getElementById('td_confidence');
    const decEl = document.getElementById('td_decision');

    if (!raw) {
        // No text detection data - show appropriate message
        if (predEl) {
            const span = predEl.querySelector('span');
            if (span) span.textContent = 'Not analyzed';
        }
        if (confEl) {
            const span = confEl.querySelector('span');
            if (span) span.textContent = 'N/A';
        }
        if (decEl) {
            const span = decEl.querySelector('span');
            if (span) span.textContent = 'Run text detection from dashboard';
        }
        return;
    }

    try {
        const res = JSON.parse(raw);
        const prediction = res.prediction || 'N/A';
        const confidence = res.confidence != null ? (Number(res.confidence).toFixed(2) + '%') : 'N/A';

        if (predEl) {
            const span = predEl.querySelector('span');
            if (span) {
                span.textContent = prediction;
                // Color based on prediction
                if (prediction.toLowerCase().includes('ai') || prediction.toLowerCase().includes('fake')) {
                    span.className = 'text-red-400 font-semibold';
                } else if (prediction.toLowerCase().includes('human') || prediction.toLowerCase().includes('real')) {
                    span.className = 'text-green-400 font-semibold';
                } else {
                    span.className = 'text-yellow-400';
                }
            }
        }
        if (confEl) {
            const span = confEl.querySelector('span');
            if (span) span.textContent = confidence;
        }

        let decisionText = 'N/A';
        if (res.prediction === 'uncertain' && res.decision) {
            const leaning = res.decision.leaning || res.decision.reason || 'unknown';
            const margin = res.decision.margin != null ? Number(res.decision.margin).toFixed(3) : 'n/a';
            decisionText = `Uncertain — leaning: ${leaning} (margin ${margin})`;
        } else if (res.decision) {
            try { decisionText = typeof res.decision === 'string' ? res.decision : JSON.stringify(res.decision); } catch (e) { decisionText = String(res.decision); }
        }

        if (decEl) {
            const span = decEl.querySelector('span');
            if (span) span.textContent = decisionText;
        }
    } catch (e) {
        console.error('Failed to parse visioNova_text_result:', e);
    }
}
