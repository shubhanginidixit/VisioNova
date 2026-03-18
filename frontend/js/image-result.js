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

    // Load image data from storage (try IndexedDB first for large files, fall back to sessionStorage)
    imageData = await VisioNovaStorage.getImageFile();
    if (!imageData) {
        imageData = VisioNovaStorage.getFile('image');
    }
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

            // Prepare JSON body for ensemble endpoint
            const base64Image = imgData.data; // Already base64 data URL or raw base64
            const jsonBody = {
                image: base64Image,
                filename: imgData.fileName || 'image.jpg',
                load_ml_models: true
            };

            const response = await fetch(`${API_BASE_URL}/api/detect-image/ensemble`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(jsonBody)
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

    let aiProb = result.ai_probability || 0;
    const humanProb = Math.max(0, 100 - aiProb);
    
    // --- 1. Probability Card (keep granular) ---
    const displayAI = aiProb;
    const displayHuman = humanProb;

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

    if (aiProb >= 80) {
        probText = 'AI-Generated Content Detected';
        probClass = 'text-xs text-accent-danger mt-2 font-medium flex items-center gap-1';
    } else if (aiProb >= 60) {
        probText = 'Likely AI — review recommended';
        probClass = 'text-xs text-accent-warning mt-2 font-medium flex items-center gap-1';
    } else if (aiProb >= 45) {
        probText = 'Uncertain: mixed signals';
        probClass = 'text-xs text-accent-warning mt-2 font-medium flex items-center gap-1';
    } else if (aiProb >= 20) {
        probText = 'Likely human-created';
        probClass = 'text-xs text-accent-success mt-2 font-medium flex items-center gap-1';
    } else {
        probText = 'Authentic signal detected';
        probClass = 'text-xs text-accent-success mt-2 font-medium flex items-center gap-1';
    }

    updateElement('probabilityText', probText);

    // Color the probability note
    const probNote = document.getElementById('probabilityNote');
    if (probNote) {
        probNote.className = probClass;
    }

    // --- 3. Analysis Mode Card ---
    const analysisMode = result.analysis_mode || 'Statistical Analysis';
    updateElement('analysisMode', analysisMode.includes('Ensemble') ? 'ML Ensemble' : 'Statistical');
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
    if (result.ai_analysis) tags.push({ name: 'XAI Analysis', color: 'primary' });

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
        findings.push({ icon: 'smart_toy', color: 'primary', text: `XAI Analysis: ${cv.verdict_description || cv.verdict}`, detail: `${Math.round(cv.combined_probability || aiProb)}% combined score` });
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
 * Render the XAI explanation — clean plain-English verdict + bullet-point reasons
 * + forensic evidence + optional Grad-CAM heatmap.
 */
function renderAIExplanation(result) {
    const textEl = document.getElementById('explanationText');
    if (!textEl) return;

    const aiAnalysis = result.ai_analysis;

    if (!aiAnalysis || !aiAnalysis.success) {
        textEl.innerHTML = `
            <div class="flex items-center gap-2 text-white/40">
                <span class="material-symbols-outlined !text-[20px]">info</span>
                <span class="text-sm">Explanation not available for this image.</span>
            </div>
        `;
        return;
    }

    const va = aiAnalysis.visual_analysis || {};
    const cv = aiAnalysis.combined_verdict || {};
    const forensic = aiAnalysis.forensic_evidence || {};

    // ── Determine overall sentiment for styling ──────────────────────────────
    const isAI = (cv.combined_probability || result.ai_probability || 50) > 50;
    const accentColor = isAI ? 'accent-danger' : 'accent-success';
    const accentIcon = isAI ? 'smart_toy' : 'verified_user';

    let html = '';

    // ── 1. Verdict sentence (summary) ───────────────────────────────────────
    const summary = va.assessment || '';
    if (summary) {
        html += `
        <div class="flex items-start gap-3 mb-4 p-4 rounded-xl border bg-${accentColor}/5 border-${accentColor}/20">
            <span class="material-symbols-outlined text-${accentColor} !text-[22px] shrink-0 mt-0.5">${accentIcon}</span>
            <div>
                <p class="text-white font-semibold text-sm leading-relaxed">${escapeHtml(summary)}</p>
                ${va.agreement_detail ? `<p class="text-white/50 text-xs mt-1">${escapeHtml(va.agreement_detail)}</p>` : ''}
            </div>
        </div>`;
    }

    // ── 2. Grad-CAM Heatmap (if available) ──────────────────────────────────
    if (va.attention_heatmap) {
        html += `
        <div class="mb-4">
            <h5 class="text-white/60 text-xs font-semibold uppercase tracking-widest mb-2">Attention Heatmap</h5>
            <div class="rounded-xl overflow-hidden border border-white/10">
                <img src="${va.attention_heatmap}" alt="Grad-CAM attention heatmap" class="w-full h-auto" />
            </div>
            <p class="text-white/40 text-[10px] mt-1">Warm areas show where the AI detector focused. Bright regions triggered the strongest detection signals.</p>
        </div>`;
    }

    // ── 3. Why? — plain-English bullet points ───────────────────────────────
    const reasoning = aiAnalysis.reasoning;
    const bullets = (reasoning && Array.isArray(reasoning.bullets)) ? reasoning.bullets : [];
    const caveat = reasoning?.caveat || null;

    if (bullets.length > 0) {
        html += `<div class="mb-4">
            <h5 class="text-white/60 text-xs font-semibold uppercase tracking-widest mb-3">Why we think this</h5>
            <ul class="space-y-2.5">
                ${bullets.map(b => `
                <li class="flex items-start gap-2.5 text-sm text-white/75 leading-relaxed">
                    <span class="text-${accentColor} shrink-0 mt-0.5">●</span>
                    <span>${escapeHtml(b)}</span>
                </li>`).join('')}
            </ul>
        </div>`;
    }

    // ── 4. Forensic Evidence Summary (compact) ──────────────────────────────
    const evidenceItems = [];
    if (forensic.metadata && forensic.metadata.status !== 'not_scanned') {
        const metaIcon = forensic.metadata.status === 'clean' ? 'check_circle' : (forensic.metadata.status === 'missing' ? 'warning' : 'error');
        const metaColor = forensic.metadata.status === 'clean' ? 'accent-success' : (forensic.metadata.status === 'missing' ? 'accent-warning' : 'accent-danger');
        evidenceItems.push({ icon: metaIcon, color: metaColor, label: 'Metadata', text: forensic.metadata.summary });
    }
    if (forensic.watermark && forensic.watermark.status !== 'not_scanned') {
        const wmColor = forensic.watermark.status === 'detected' ? 'accent-danger' : 'accent-success';
        evidenceItems.push({ icon: 'fingerprint', color: wmColor, label: 'Watermark', text: forensic.watermark.summary });
    }
    if (forensic.c2pa && forensic.c2pa.status !== 'not_scanned') {
        const c2Color = forensic.c2pa.status === 'ai_confirmed' ? 'accent-danger' : (forensic.c2pa.status === 'authentic' ? 'accent-success' : 'white/50');
        evidenceItems.push({ icon: 'verified', color: c2Color, label: 'C2PA', text: forensic.c2pa.summary });
    }

    if (evidenceItems.length > 0) {
        html += `<div class="mb-4">
            <h5 class="text-white/60 text-xs font-semibold uppercase tracking-widest mb-2">Forensic Evidence</h5>
            <div class="space-y-1.5">
                ${evidenceItems.map(e => `
                <div class="flex items-start gap-2 p-2 bg-white/[0.03] rounded-lg">
                    <span class="material-symbols-outlined text-${e.color} !text-[16px] shrink-0 mt-0.5">${e.icon}</span>
                    <div class="min-w-0">
                        <span class="text-white/50 text-[10px] font-bold uppercase">${e.label}</span>
                        <p class="text-white/60 text-xs leading-relaxed">${escapeHtml(e.text)}</p>
                    </div>
                </div>`).join('')}
            </div>
        </div>`;
    }

    // ── 5. Caveat (only for uncertain / split results) ───────────────────────
    if (caveat) {
        html += `
        <div class="mt-3 flex items-start gap-2 p-3 bg-accent-warning/5 border border-accent-warning/20 rounded-xl">
            <span class="material-symbols-outlined text-accent-warning !text-[18px] shrink-0 mt-0.5">info</span>
            <p class="text-white/60 text-xs leading-relaxed">${escapeHtml(caveat)}</p>
        </div>`;
    }

    // ── 6. Verdict chip at the bottom ────────────────────────────────────────
    const verdictLabel = cv.verdict_description || (isAI ? 'AI-Generated' : 'Authentic');
    html += `
    <div class="mt-4 pt-3 border-t border-white/10 flex items-center justify-between">
        <span class="text-white/40 text-xs">Final verdict</span>
        <span class="px-3 py-1 rounded-full text-xs font-bold bg-${accentColor}/15 text-${accentColor} border border-${accentColor}/25">
            ${escapeHtml(verdictLabel)}
        </span>
    </div>`;

    textEl.innerHTML = html;

    // Auto-expand the explanation panel
    document.getElementById('explanationContent')?.classList.remove('hidden');
    document.getElementById('explanationChevron')?.classList.add('rotated');
}



/**
 * Render detection models detail in the collapsible panel.
 * Uses the model_breakdown from XAI analysis for enriched per-model cards.
 */
function renderModelsDetail(result) {
    const container = document.getElementById('modelsList');
    if (!container) return;

    const models = [];
    const aiAnalysis = result.ai_analysis;
    const modelBreakdown = aiAnalysis?.visual_analysis?.model_breakdown || [];

    // Primary: show real model cards from XAI ensemble analysis
    if (modelBreakdown.length > 0) {
        modelBreakdown.forEach(m => {
            const isAI = m.flagged_as_ai;
            models.push({
                name: m.name || m.key,
                icon: isAI ? 'smart_toy' : 'verified_user',
                description: m.interpretation || m.specialty,
                detail: `${m.accuracy ? 'Acc: ' + m.accuracy + ' • ' : ''}Score: ${Math.round(m.score)}%`,
                specialty: m.specialty,
                how_it_works: m.how_it_works || '',
                evidence: m.evidence || '',
                status: isAI ? 'AI Detected' : 'Human',
                statusColor: isAI ? 'accent-danger' : 'accent-success',
                score: m.score,
            });
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

    // Update subtitle
    updateElement('modelsSubtitle', `${models.length} model${models.length !== 1 ? 's' : ''} contributed to this analysis`);

    // Render with enriched cards for ML models
    container.innerHTML = models.map(m => {
        // Build expanded card for ML models with how_it_works and evidence
        let extraHTML = '';
        if (m.how_it_works || m.evidence) {
            extraHTML = `<div class="mt-2 space-y-1.5">`;
            if (m.how_it_works) {
                extraHTML += `<p class="text-white/40 text-xs leading-relaxed"><span class="text-white/50 font-medium">How:</span> ${escapeHtml(m.how_it_works)}</p>`;
            }
            if (m.evidence) {
                extraHTML += `<p class="text-white/40 text-xs leading-relaxed"><span class="text-white/50 font-medium">Evidence:</span> ${escapeHtml(m.evidence)}</p>`;
            }
            extraHTML += `</div>`;
        }

        // Score bar for ML models
        let scoreBar = '';
        if (m.score !== undefined) {
            const barColor = m.score > 50 ? '#FF4A4A' : '#00D991';
            scoreBar = `
                <div class="mt-2 w-full bg-black/30 rounded-full h-1.5">
                    <div class="h-1.5 rounded-full transition-all duration-700" style="width: ${Math.round(m.score)}%; background: ${barColor}"></div>
                </div>`;
        }

        return `
        <div class="flex flex-col gap-1 p-3 bg-white/5 rounded-xl hover:bg-white/[0.07] transition-colors">
            <div class="flex items-start gap-3">
                <div class="p-2 rounded-lg bg-white/5 shrink-0">
                    <span class="material-symbols-outlined text-white/60 !text-[20px]">${m.icon}</span>
                </div>
                <div class="flex-1 min-w-0">
                    <div class="flex items-center gap-2 flex-wrap">
                        <p class="text-white text-sm font-medium">${escapeHtml(m.name)}</p>
                        <span class="px-1.5 py-0.5 rounded text-[10px] font-bold bg-${m.statusColor}/10 text-${m.statusColor}">${m.status}</span>
                    </div>
                    <p class="text-white/50 text-xs mt-0.5">${escapeHtml(m.description)}</p>
                    ${m.detail ? `<p class="text-white/40 text-[10px] mt-0.5">${escapeHtml(m.detail)}</p>` : ''}
                    ${scoreBar}
                    ${extraHTML}
                </div>
            </div>
        </div>`;
    }).join('');
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
