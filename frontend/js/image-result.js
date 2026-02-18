/**
 * VisioNova Image Analysis - Simplified Result Page
 * Shows: verdict, probability, image preview, key findings, watermark, AI explanation, models used
 * Follows the text-result.js pattern: simple, clean, model-focused
 */

// ============================================================================
// Configuration & Constants
// ============================================================================

const API_BASE_URL = 'http://localhost:5000';

// ============================================================================
// State Management
// ============================================================================

let currentResult = null;   // Store current analysis result
let imageData = null;       // Store original image data (base64 or URL)

// ============================================================================
// Initialization
// ============================================================================

document.addEventListener('DOMContentLoaded', async function () {
    console.log('[ImageResult] Page loaded, checking for image data...');

    // Warn if accessed via file:// protocol (API calls still work via CORS)
    if (window.location.protocol === 'file:') {
        console.warn('[ImageResult] WARNING: Page opened from file system. API calls will use CORS.');
    }

    // Setup collapsible panels
    setupCollapsiblePanels();

    // Setup action buttons
    setupActionButtons();

    // Load image data from storage
    imageData = VisioNovaStorage.getFile('image');
    console.log('[ImageResult] Image data from storage:', imageData ? 'found' : 'not found');

    if (imageData && imageData.data) {
        // Show the image preview
        displayImagePreview(imageData.data, imageData.fileName);

        // Process analysis results
        await processImageAnalysis(imageData);
    } else {
        // No image data — show empty state
        showNoImageState();
    }
});

// ============================================================================
// Image Preview Display
// ============================================================================

/**
 * Show the uploaded image in the preview area.
 * imageDataStr can be a base64 data URL or a regular URL.
 */
function displayImagePreview(imageDataStr, fileName) {
    const previewImage = document.getElementById('previewImage');
    const placeholder = document.getElementById('imagePreviewPlaceholder');

    if (!previewImage) return;

    // Set the image source
    previewImage.src = imageDataStr;
    previewImage.alt = fileName || 'Analyzed image';

    // Show image, hide placeholder
    previewImage.classList.remove('hidden');
    if (placeholder) placeholder.classList.add('hidden');

    // Also set fullscreen image
    const fullscreenImg = document.getElementById('fullscreenImage');
    if (fullscreenImg) fullscreenImg.src = imageDataStr;
}

/**
 * Show a state when no image was provided.
 */
function showNoImageState() {
    const placeholder = document.getElementById('imagePreviewPlaceholder');
    if (placeholder) {
        placeholder.innerHTML = `
            <span class="material-symbols-outlined text-6xl opacity-50">image_not_supported</span>
            <p class="text-sm">No image provided for analysis</p>
            <a href="homepage.html" class="text-primary hover:underline text-sm">Upload an image to analyze</a>
        `;
    }
    updateElement('aiScore', '--');
    updateElement('verdictBadge', 'No Data');
    updateElement('verdictDescription', 'Upload an image from the homepage to get started');
}

// ============================================================================
// Analysis Processing
// ============================================================================

/**
 * Main analysis function.
 * Tries to get results from sessionStorage first (stored by AnalysisDashboard),
 * falls back to calling the API directly.
 */
async function processImageAnalysis(imgData) {
    console.log('[ImageResult] Processing image analysis...');

    try {
        // Step 1: Check if AnalysisDashboard already stored results
        const storedResult = sessionStorage.getItem('visioNova_image_result');
        let result;

        if (storedResult) {
            console.log('[ImageResult] Found stored analysis result from AnalysisDashboard');
            result = JSON.parse(storedResult);
        } else {
            // Step 2: Fallback — call the API directly
            console.log('[ImageResult] No stored result found, calling API directly...');
            showLoadingState();

            // Prepare the image data for API
            const formData = new FormData();

            if (imgData.mimeType === 'url') {
                // URL-based image
                formData.append('url', imgData.data);
            } else {
                // Base64 image — convert to blob
                const base64Data = imgData.data.split(',')[1] || imgData.data;
                const binaryStr = atob(base64Data);
                const bytes = new Uint8Array(binaryStr.length);
                for (let i = 0; i < binaryStr.length; i++) {
                    bytes[i] = binaryStr.charCodeAt(i);
                }
                const blob = new Blob([bytes], { type: imgData.mimeType || 'image/jpeg' });
                formData.append('image', blob, imgData.fileName || 'image.jpg');
            }

            // Include watermark analysis
            formData.append('include_watermark', 'true');
            formData.append('include_ai_analysis', 'true');

            const response = await fetch(`${API_BASE_URL}/api/detect-image`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`API returned ${response.status}: ${response.statusText}`);
            }

            result = await response.json();
        }

        // Step 3: Update the UI with the result
        if (result && result.success !== false) {
            currentResult = result;
            updateUI(result);
        } else {
            const errorMsg = result?.error || 'Analysis failed';
            displayError(errorMsg);
        }

    } catch (error) {
        console.error('[ImageResult] Analysis error:', error);
        displayError(`Analysis failed: ${error.message}`);
    }
}

// ============================================================================
// UI Update — The main function that populates everything
// ============================================================================

/**
 * Update all UI elements with the analysis result.
 * This is the core function — it reads the API response and fills in:
 * - Verdict card (score, badge, bar)
 * - Human vs AI probability bars
 * - Analysis mode card
 * - Key findings list
 * - Watermark detection status
 * - AI explanation text
 * - Detection models list
 */
function updateUI(result) {
    console.log('[ImageResult] Updating UI with result:', result);

    const aiProb = result.ai_probability || 0;
    const humanProb = Math.max(0, 100 - aiProb);

    // --- 1. Probability Card (restored) ---
    // Hybrid "Winner takes all" logic:
    // Only snap to 100% if confidence is high (> 70%).
    // Otherwise show raw probabilities to avoid false positives on marginal cases.
    let displayAI = aiProb;
    let displayHuman = humanProb;

    if (aiProb >= 70) {
        displayAI = 100;
        displayHuman = 0;
    } else if (humanProb >= 70) {
        displayHuman = 100;
        displayAI = 0;
    }
    // Else: keep raw values (e.g., 55% vs 45%)

    // Update Analysis Date (kept from header)
    updateElement('analysisDate', `Analyzed ${formatAnalysisDate(new Date())}`);

    // Update Probability Bars
    const humanBar = document.getElementById('humanBar');
    const aiBar = document.getElementById('aiBar');
    if (humanBar) setTimeout(() => { humanBar.style.height = `${displayHuman}%`; }, 200);
    if (aiBar) setTimeout(() => { aiBar.style.height = `${displayAI}%`; }, 200);

    updateElement('humanPercent', `${Math.round(displayHuman)}%`);
    updateElement('aiPercent', `${Math.round(displayAI)}%`);

    // Probability note text (hybrid)
    let probText = '';
    let probClass = '';

    if (displayAI === 100) {
        probText = 'AI-Generated Content Detected';
        probClass = 'text-xs text-accent-danger mt-2 font-medium flex items-center gap-1';
    } else if (displayHuman === 100) {
        probText = 'Human-Created Content Detected';
        probClass = 'text-xs text-accent-success mt-2 font-medium flex items-center gap-1';
    } else {
        // Uncertain / Mixed
        probText = 'Inconclusive: Mixed patterns detected';
        probClass = 'text-xs text-accent-warning mt-2 font-medium flex items-center gap-1';
    }

    updateElement('probabilityText', probText);

    // Color the probability note
    const probNote = document.getElementById('probabilityNote');
    if (probNote) {
        probNote.className = probClass;
    }

    // --- 3. Analysis Mode Card ---
    const analysisMode = result.analysis_mode || 'Statistical Analysis';
    updateElement('analysisMode', analysisMode.includes('ML') ? 'ML Ensemble' : 'Statistical');
    updateElement('analysisModeInfo', analysisMode);

    // Models used tags
    renderModelTags(result);

    // --- 4. Image Info Badges ---
    if (result.dimensions) {
        updateElement('imageDimensions', `${result.dimensions.width} × ${result.dimensions.height}`);
    }
    if (result.file_size) {
        const sizeKB = Math.round(result.file_size / 1024);
        updateElement('imageSize', sizeKB > 1024 ? `${(sizeKB / 1024).toFixed(1)} MB` : `${sizeKB} KB`);
    }

    // --- 5. Key Findings ---
    renderKeyFindings(result);

    // --- 6. Watermark Detection ---
    renderWatermarkResult(result);

    // --- 7. AI Explanation ---
    renderAIExplanation(result);

    // --- 8. Detection Models Detail ---
    renderModelsDetail(result);
}

// ============================================================================
// Rendering Functions
// ============================================================================

/**
 * Render small model tags in the Analysis Mode card.
 */
function renderModelTags(result) {
    const container = document.getElementById('modelsUsedTags');
    if (!container) return;

    const tags = [];

    // ML models
    if (result.ml_prediction) tags.push({ name: 'ML Model', color: 'primary' });
    if (result.analysis_mode && result.analysis_mode.includes('Statistical')) tags.push({ name: 'Statistical', color: 'primary' });

    // Semantic
    if (result.semantic_analysis) tags.push({ name: 'Semantic AI', color: 'accent-warning' });

    // C2PA
    if (result.c2pa || result.content_credentials) tags.push({ name: 'C2PA', color: 'accent-success' });

    // Watermark
    if (result.watermark) tags.push({ name: 'Watermark', color: 'accent-success' });

    // AI Analysis (Groq Vision)
    if (result.ai_analysis) tags.push({ name: 'Groq Vision', color: 'primary' });

    // If no specific tags, show a default
    if (tags.length === 0) tags.push({ name: 'Auto-detect', color: 'primary' });

    container.innerHTML = tags.map(tag =>
        `<span class="px-2 py-0.5 rounded text-[10px] font-bold bg-${tag.color}/10 text-${tag.color} border border-${tag.color}/20">${tag.name}</span>`
    ).join('');
}

/**
 * Render key findings — the most important signals from the analysis.
 */
function renderKeyFindings(result) {
    const container = document.getElementById('keyFindings');
    const countEl = document.getElementById('findingsCount');
    if (!container) return;

    const findings = [];
    const aiProb = result.ai_probability || 0;

    // Finding 1: Overall verdict
    if (aiProb >= 80) {
        findings.push({ icon: 'warning', color: 'accent-danger', text: 'High probability of AI generation detected', detail: `${Math.round(aiProb)}% AI probability` });
    } else if (aiProb >= 50) {
        findings.push({ icon: 'info', color: 'accent-warning', text: 'Moderate AI generation indicators found', detail: `${Math.round(aiProb)}% AI probability` });
    } else {
        findings.push({ icon: 'check_circle', color: 'accent-success', text: 'Image appears to be human-created', detail: `${Math.round(100 - aiProb)}% human probability` });
    }

    // Finding 2: ML model prediction
    if (result.ml_prediction) {
        const mlConf = result.ml_prediction.confidence || 0;
        const mlLabel = result.ml_prediction.label || 'unknown';
        if (mlLabel.toLowerCase().includes('ai') || mlLabel.toLowerCase().includes('fake')) {
            findings.push({ icon: 'psychology', color: 'accent-danger', text: `ML model predicts AI-generated`, detail: `${Math.round(mlConf)}% confidence` });
        } else {
            findings.push({ icon: 'psychology', color: 'accent-success', text: `ML model predicts human-created`, detail: `${Math.round(mlConf)}% confidence` });
        }
    }

    // Finding 3: Semantic analysis
    if (result.semantic_analysis && result.semantic_analysis.success) {
        const plausibility = result.semantic_analysis.plausibility_score || 100;
        if (plausibility < 70) {
            findings.push({ icon: 'visibility', color: 'accent-warning', text: 'Visual implausibilities detected', detail: `${plausibility}% plausibility score` });
        } else {
            findings.push({ icon: 'visibility', color: 'accent-success', text: 'Image appears visually plausible', detail: `${plausibility}% plausibility score` });
        }
    }

    // Finding 4: C2PA / Content Credentials
    if (result.c2pa && result.c2pa.has_content_credentials) {
        if (result.c2pa.is_ai_generated) {
            findings.push({ icon: 'verified', color: 'accent-danger', text: `C2PA confirms AI generation`, detail: result.c2pa.ai_generator || 'Unknown tool' });
        } else {
            findings.push({ icon: 'verified', color: 'accent-success', text: 'Content Credentials verified', detail: 'Provenance data found' });
        }
    }

    // Finding 5: Watermark
    if (result.watermark && result.watermark.watermark_detected) {
        findings.push({ icon: 'fingerprint', color: 'accent-danger', text: 'AI watermark detected in image', detail: `Source: ${result.watermark.source || 'Unknown'}` });
    }

    // Finding 6: AI analysis verdict
    if (result.ai_analysis && result.ai_analysis.success && result.ai_analysis.combined_verdict) {
        const cv = result.ai_analysis.combined_verdict;
        findings.push({ icon: 'smart_toy', color: 'primary', text: `Groq Vision: ${cv.verdict_description || cv.verdict}`, detail: `${Math.round(cv.combined_probability || aiProb)}% combined score` });
    }

    // Update count
    if (countEl) countEl.textContent = `${findings.length} finding${findings.length !== 1 ? 's' : ''} detected`;

    // Render
    container.innerHTML = findings.map(f => `
        <div class="flex items-start gap-3 p-3 bg-white/5 rounded-xl hover:bg-white/8 transition-colors">
            <span class="material-symbols-outlined text-${f.color} !text-[20px] mt-0.5 shrink-0">${f.icon}</span>
            <div class="flex-1 min-w-0">
                <p class="text-white text-sm font-medium">${escapeHtml(f.text)}</p>
                <p class="text-white/50 text-xs mt-0.5">${escapeHtml(f.detail)}</p>
            </div>
        </div>
    `).join('');
}

/**
 * Render the AI watermark detection result.
 */
function renderWatermarkResult(result) {
    const container = document.getElementById('watermarkResult');
    const badge = document.getElementById('watermarkStatusBadge');
    if (!container) return;

    const watermark = result.watermark;

    if (!watermark) {
        // Watermark analysis not performed
        if (badge) {
            badge.textContent = 'Not Scanned';
            badge.className = 'px-2.5 py-1 rounded-full text-xs font-bold bg-white/5 text-white/50 border border-white/10';
        }
        container.innerHTML = `
            <div class="flex items-center gap-2 p-3 bg-white/5 rounded-xl">
                <span class="material-symbols-outlined text-white/40 !text-[20px]">info</span>
                <span class="text-white/60 text-sm">Watermark analysis was not included in this scan</span>
            </div>
        `;
        return;
    }

    if (watermark.watermark_detected) {
        // Watermark found — this is significant
        if (badge) {
            badge.textContent = 'Detected';
            badge.className = 'px-2.5 py-1 rounded-full text-xs font-bold bg-accent-danger/10 text-accent-danger border border-accent-danger/20';
        }
        container.innerHTML = `
            <div class="flex items-center gap-2 p-3 bg-accent-danger/10 rounded-xl border border-accent-danger/20">
                <span class="material-symbols-outlined text-accent-danger !text-[20px]">fingerprint</span>
                <div>
                    <p class="text-white text-sm font-medium">AI Watermark Found</p>
                    <p class="text-white/60 text-xs mt-0.5">Source: ${escapeHtml(watermark.source || watermark.detected_watermark || 'Unknown AI tool')} • Confidence: ${Math.round(watermark.confidence || 0)}%</p>
                </div>
            </div>
        `;
    } else {
        // No watermark found
        if (badge) {
            badge.textContent = 'Clear';
            badge.className = 'px-2.5 py-1 rounded-full text-xs font-bold bg-accent-success/10 text-accent-success border border-accent-success/20';
        }
        container.innerHTML = `
            <div class="flex items-center gap-2 p-3 bg-accent-success/10 rounded-xl border border-accent-success/20">
                <span class="material-symbols-outlined text-accent-success !text-[20px]">check_circle</span>
                <div>
                    <p class="text-white text-sm font-medium">No AI Watermark Found</p>
                    <p class="text-white/60 text-xs mt-0.5">Scanned with ${watermark.methods_count || 'multiple'} detection methods</p>
                </div>
            </div>
        `;
    }
}

/**
 * Render the AI explanation from Groq Vision.
 */
function renderAIExplanation(result) {
    const textEl = document.getElementById('explanationText');
    if (!textEl) return;

    const aiAnalysis = result.ai_analysis;

    if (!aiAnalysis || !aiAnalysis.success) {
        textEl.innerHTML = `
            <div class="flex items-center gap-2 text-white/40">
                <span class="material-symbols-outlined !text-[20px]">info</span>
                <span>AI explanation not available for this analysis. This feature uses Groq Vision to provide a plain-English explanation of the detection.</span>
            </div>
        `;
        return;
    }

    // Build formatted explanation
    let html = '';

    // Visual analysis
    if (aiAnalysis.visual_analysis) {
        const va = aiAnalysis.visual_analysis;
        if (va.assessment) {
            html += `<div class="mb-4">
                <h5 class="text-white font-semibold text-sm mb-2 flex items-center gap-1">
                    <span class="material-symbols-outlined !text-[16px] text-primary">visibility</span>
                    Visual Assessment
                </h5>
                <p class="text-white/70 text-sm leading-relaxed">${escapeHtml(va.assessment)}</p>
            </div>`;
        }

        // Anomalies
        if (va.anomalies && va.anomalies.length > 0) {
            html += `<div class="mb-4">
                <h5 class="text-white font-semibold text-sm mb-2 flex items-center gap-1">
                    <span class="material-symbols-outlined !text-[16px] text-accent-warning">report</span>
                    Anomalies Detected
                </h5>
                <ul class="space-y-1">
                    ${va.anomalies.map(a => `<li class="text-white/60 text-sm flex items-start gap-2">
                        <span class="text-accent-warning mt-1">•</span>
                        <span>${escapeHtml(a)}</span>
                    </li>`).join('')}
                </ul>
            </div>`;
        }
    }

    // Reasoning
    if (aiAnalysis.reasoning) {
        html += `<div class="mb-4">
            <h5 class="text-white font-semibold text-sm mb-2 flex items-center gap-1">
                <span class="material-symbols-outlined !text-[16px] text-primary">psychology</span>
                Reasoning
            </h5>
            <p class="text-white/70 text-sm leading-relaxed">${escapeHtml(typeof aiAnalysis.reasoning === 'string' ? aiAnalysis.reasoning : JSON.stringify(aiAnalysis.reasoning))}</p>
        </div>`;
    }

    // Combined verdict
    if (aiAnalysis.combined_verdict) {
        const cv = aiAnalysis.combined_verdict;
        html += `<div class="p-3 bg-white/5 rounded-xl">
            <h5 class="text-white font-semibold text-sm mb-1">Combined Verdict</h5>
            <p class="text-white/70 text-sm">${escapeHtml(cv.verdict_description || cv.verdict)}</p>
            <p class="text-white/50 text-xs mt-1">Combined probability: ${Math.round(cv.combined_probability || 0)}%</p>
        </div>`;
    }

    // Fallback: if no structured data, show raw explanation
    if (!html && aiAnalysis.explanation) {
        html = `<p class="text-white/70 text-sm leading-relaxed">${escapeHtml(typeof aiAnalysis.explanation === 'string' ? aiAnalysis.explanation : JSON.stringify(aiAnalysis.explanation))}</p>`;
    }

    textEl.innerHTML = html || '<p class="text-white/40 text-sm">No detailed explanation available.</p>';

    // Auto-expand if there's an explanation
    if (html) {
        document.getElementById('explanationContent')?.classList.remove('hidden');
        document.getElementById('explanationChevron')?.classList.add('rotated');
    }
}

/**
 * Render detection models detail in the collapsible panel.
 */
function renderModelsDetail(result) {
    const container = document.getElementById('modelsList');
    if (!container) return;

    const models = [];

    // Statistical Analysis (always runs)
    if (result.analysis_scores) {
        models.push({
            name: 'Statistical Analysis',
            icon: 'bar_chart',
            description: 'Color distribution, noise patterns, edge analysis, texture consistency',
            status: 'Completed',
            statusColor: 'accent-success'
        });
    }

    // ML Prediction
    if (result.ml_prediction) {
        models.push({
            name: 'ML Ensemble (DIRE + NYUAD)',
            icon: 'model_training',
            description: `Prediction: ${result.ml_prediction.label || 'unknown'} • Confidence: ${Math.round(result.ml_prediction.confidence || 0)}%`,
            status: 'Completed',
            statusColor: 'accent-success'
        });
    }

    // Semantic Analysis
    if (result.semantic_analysis) {
        const sa = result.semantic_analysis;
        models.push({
            name: 'Semantic Plausibility (Groq LLaVA)',
            icon: 'visibility',
            description: sa.success
                ? `Plausibility: ${sa.plausibility_score || 'N/A'}% • ${sa.summary || 'Visual analysis complete'}`
                : `Error: ${sa.error || 'Analysis failed'}`,
            status: sa.success ? 'Completed' : 'Error',
            statusColor: sa.success ? 'accent-success' : 'accent-danger'
        });
    }

    // C2PA
    if (result.c2pa || result.content_credentials) {
        const c2pa = result.c2pa || result.content_credentials;
        models.push({
            name: 'Content Credentials (C2PA)',
            icon: 'verified_user',
            description: c2pa.has_content_credentials
                ? `Credentials found • ${c2pa.is_ai_generated ? 'AI generated' : 'Authentic source'}`
                : 'No content credentials found in image',
            status: 'Completed',
            statusColor: 'accent-success'
        });
    }

    // Watermark
    if (result.watermark) {
        models.push({
            name: 'Invisible Watermark Scanner',
            icon: 'fingerprint',
            description: result.watermark.watermark_detected
                ? `AI watermark detected: ${result.watermark.source || 'Unknown'}`
                : 'No AI watermarks found',
            status: 'Completed',
            statusColor: 'accent-success'
        });
    }

    // AI Explanation (Groq Vision)
    if (result.ai_analysis) {
        models.push({
            name: 'Groq Vision (Image Explainer)',
            icon: 'smart_toy',
            description: result.ai_analysis.success
                ? 'AI-powered visual analysis and explanation generated'
                : `Error: ${result.ai_analysis.error || 'Analysis unavailable'}`,
            status: result.ai_analysis.success ? 'Completed' : 'Error',
            statusColor: result.ai_analysis.success ? 'accent-success' : 'accent-danger'
        });
    }

    // Update subtitle
    updateElement('modelsSubtitle', `${models.length} model${models.length !== 1 ? 's' : ''} contributed to this analysis`);

    // Render
    container.innerHTML = models.map(m => `
        <div class="flex items-start gap-3 p-3 bg-white/5 rounded-xl">
            <div class="p-2 rounded-lg bg-white/5 shrink-0">
                <span class="material-symbols-outlined text-white/60 !text-[20px]">${m.icon}</span>
            </div>
            <div class="flex-1 min-w-0">
                <div class="flex items-center gap-2">
                    <p class="text-white text-sm font-medium">${escapeHtml(m.name)}</p>
                    <span class="px-1.5 py-0.5 rounded text-[10px] font-bold bg-${m.statusColor}/10 text-${m.statusColor}">${m.status}</span>
                </div>
                <p class="text-white/50 text-xs mt-0.5">${escapeHtml(m.description)}</p>
            </div>
        </div>
    `).join('');
}

// ============================================================================
// UI Helpers
// ============================================================================

/**
 * Show loading state while analysis is in progress.
 */
function showLoadingState() {
    updateElement('aiScore', '...');
    updateElement('verdictBadge', 'Analyzing');
    updateElement('verdictDescription', 'Running AI detection models...');
}

/**
 * Display an error message in the key findings area.
 */
function displayError(message) {
    const container = document.getElementById('keyFindings');
    if (container) {
        container.innerHTML = `
            <div class="flex items-start gap-3 p-4 bg-accent-danger/10 rounded-xl border border-accent-danger/20">
                <span class="material-symbols-outlined text-accent-danger !text-[24px] shrink-0">error</span>
                <div>
                    <p class="text-white font-medium text-sm">Analysis Error</p>
                    <p class="text-white/60 text-xs mt-1">${escapeHtml(message)}</p>
                </div>
            </div>
        `;
    }
    updateElement('findingsCount', 'Error occurred');
    updateElement('aiScore', '--');
    updateElement('verdictBadge', 'Error');
    updateElement('verdictDescription', 'Analysis could not be completed');
}

/**
 * Get verdict badge configuration based on verdict and probability.
 */
function getVerdictBadgeConfig(verdict, aiProb) {
    // Binary badge logic to match "winner takes all" display
    if (aiProb > 50) {
        return { text: 'AI-Generated', classes: 'bg-accent-danger/20 text-accent-danger border-accent-danger/20' };
    } else {
        return { text: 'Human-Created', classes: 'bg-accent-success/20 text-accent-success border-accent-success/20' };
    }
}

// ============================================================================
// Collapsible Panels & Action Buttons
// ============================================================================

/**
 * Setup toggle behavior for the AI Explanation and Models panels.
 */
function setupCollapsiblePanels() {
    // AI Explanation panel
    const explToggle = document.getElementById('explanationToggle');
    const explContent = document.getElementById('explanationContent');
    const explChevron = document.getElementById('explanationChevron');
    if (explToggle && explContent) {
        explToggle.addEventListener('click', () => {
            explContent.classList.toggle('hidden');
            explChevron?.classList.toggle('rotated');
        });
    }

    // Models panel
    const modelsToggle = document.getElementById('modelsToggle');
    const modelsContent = document.getElementById('modelsContent');
    const modelsChevron = document.getElementById('modelsChevron');
    if (modelsToggle && modelsContent) {
        modelsToggle.addEventListener('click', () => {
            modelsContent.classList.toggle('hidden');
            modelsChevron?.classList.toggle('rotated');
        });
    }
}

/**
 * Setup the Re-analyze and Export buttons.
 */
function setupActionButtons() {
    // Re-analyze button
    const reanalyzeBtn = document.getElementById('reanalyzeBtn');
    if (reanalyzeBtn) {
        reanalyzeBtn.addEventListener('click', async () => {
            if (!imageData) return;
            // Clear stored result so we call the API fresh
            sessionStorage.removeItem('visioNova_image_result');
            showLoadingState();
            await processImageAnalysis(imageData);
        });
    }

    // Export button (placeholder)
    const exportBtn = document.getElementById('exportPdfBtn');
    if (exportBtn) {
        exportBtn.addEventListener('click', () => {
            alert('Export feature coming soon!');
        });
    }
}

// ============================================================================
// Fullscreen Viewer
// ============================================================================

function openFullscreenViewer() {
    const modal = document.getElementById('fullscreenModal');
    if (modal) modal.classList.remove('hidden');
    if (modal) modal.classList.add('flex');
}

function closeFullscreenViewer() {
    const modal = document.getElementById('fullscreenModal');
    if (modal) modal.classList.add('hidden');
    if (modal) modal.classList.remove('flex');
}

// ============================================================================
// Generic Helpers
// ============================================================================

/**
 * Safely update a DOM element's text content by ID.
 */
function updateElement(id, text) {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
}

/**
 * Escape HTML to prevent XSS.
 */
function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = String(text);
    return div.innerHTML;
}

/**
 * Format a date for display.
 */
function formatAnalysisDate(date) {
    const now = new Date();
    const diff = now - date;
    if (diff < 60000) return 'just now';
    if (diff < 3600000) return `${Math.floor(diff / 60000)} min ago`;
    return date.toLocaleString();
}
