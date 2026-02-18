/**
 * VisioNova Image Analysis - Dynamic Tab-Based Implementation
 * Follows the fact-check.js pattern for modular content rendering
 */

// ============================================================================
// Configuration & Constants
// ============================================================================

const API_BASE_URL = 'http://localhost:5000';

// ============================================================================
// State Management
// ============================================================================

let currentResult = null;           // Store current analysis result
let currentTab = 'overview';        // Current active tab
let currentVisualization = 'original';  // Current visualization view
let imageData = null;               // Store original image data

// ============================================================================
// Initialization
// ============================================================================

document.addEventListener('DOMContentLoaded', async function () {
    console.log('[ImageResult] Page loaded, checking for image data...');

    // Check if accessed via file:// protocol
    if (window.location.protocol === 'file:') {
        console.error('[ImageResult] ERROR: Page opened from file system. CORS will block API calls.');
        displayError('Please access this page through the Flask server (http://localhost:5000) instead of opening the HTML file directly.');
        return;
    }

    // Initialize tab system
    initTabs();

    // Initialize button handlers
    initButtonHandlers();

    // Load image data from storage
    imageData = VisioNovaStorage.getFile('image');
    console.log('[ImageResult] Image data found:', imageData ? 'Yes' : 'No');

    if (imageData) {
        await processImageAnalysis(imageData);
    } else {
        console.log('[ImageResult] No image data found in storage');
        displayError('No image found. Please upload an image from the homepage.');
    }
});

/**
 * Initialize tab system
 */
function initTabs() {
    UI.initTabs('.tab-btn', (tabId) => {
        currentTab = tabId;
        if (currentResult) {
            renderTabContent(currentResult, tabId);
        }
    });
}

/**
 * Render content based on the selected tab
 */
function renderTabContent(result, tabName) {
    const tabContent = document.getElementById('tab-content');
    if (!tabContent) return;

    switch (tabName) {
        case 'overview':
            tabContent.innerHTML = buildOverviewHTML(result);
            break;
        case 'technical':
            tabContent.innerHTML = buildTechnicalHTML(result);
            break;
        case 'metadata':
            tabContent.innerHTML = buildMetadataHTML(result);
            break;
        case 'ai-analysis':
            tabContent.innerHTML = buildAIAnalysisHTML(result);
            break;
        default:
            tabContent.innerHTML = buildOverviewHTML(result);
    }

    // Re-attach event listeners after rendering
    attachEventListeners();
}

// ============================================================================
// Tab Content Builders
// ============================================================================

/**
 * Build Overview Tab Content
 */
function buildOverviewHTML(result) {
    const aiProbability = result.ai_probability || 50;
    let verdict = result.verdict || (aiProbability > 50 ? 'LIKELY_AI' : 'LIKELY_REAL');

    // If watermark detected or C2PA confirms AI, authenticity = 0 and verdict = AI_GENERATED
    let authenticityScore = Math.round(100 - aiProbability);
    if (result.watermark?.watermark_detected || result.content_credentials?.is_ai_generated) {
        authenticityScore = 0;
        verdict = 'AI_GENERATED';
    }

    const verdictConfig = getVerdictConfig(verdict);

    return `
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <!-- Left Column: Image Viewer & Visualizations -->
            <div class="lg:col-span-2 space-y-6">
                ${buildImageViewerCard(result)}
                ${buildQuickMetricsGrid(result)}
            </div>
            
            <!-- Right Column: Score & Summary -->
            <div class="space-y-6">
                ${buildScoreCard(result, authenticityScore, verdictConfig)}
                ${buildVerificationSummaryCard(result)}
                ${buildQuickActionsCard()}
            </div>
        </div>
        
        <!-- Full Width: Key Findings -->
        <div class="mt-6">
            ${buildKeyFindingsCard(result)}
        </div>
    `;
}

/**
 * Build Technical Analysis Tab Content
 */
function buildTechnicalHTML(result) {
    return `
        <div class="space-y-6">
            ${buildDetectionMethodsCard(result)}
            ${buildAdvancedDetectorsCard(result)}
            ${buildForensicAnalysisCard(result)}
            ${buildELAAnalysisCard(result)}
            ${buildNoiseAnalysisCard(result)}
            ${buildWatermarkDetectionCard(result)}
        </div>
    `;
}

/**
 * Build Metadata Deep-Dive Tab Content
 */
function buildMetadataHTML(result) {
    return `
        <div class="space-y-6">
            ${buildMetadataOverviewCard(result)}
            ${buildEXIFDataCard(result)}
            ${buildWatermarkMetadataCard(result)}
            ${buildC2PACredentialsCard(result)}
            ${buildFilePropertiesCard(result)}
        </div>
    `;
}

/**
 * Build AI Explanation Tab Content
 */
function buildAIAnalysisHTML(result) {
    return `
        <div class="space-y-6">
            ${buildAIExplanationCard(result)}
            ${buildVisualAnalysisCard(result)}
            ${buildConfidenceBreakdownCard(result)}
        </div>
    `;
}

// ============================================================================
// Component Builders - Overview Tab
// ============================================================================

/**
 * Build Image Viewer Card with visualization controls
 */
function buildImageViewerCard(result) {
    const visualizations = [
        { id: 'original', label: 'Original', icon: 'image' },
        { id: 'ela', label: 'ELA Analysis', icon: 'gradient' },
        { id: 'heatmap', label: 'AI Heatmap', icon: 'blur_on' },
        { id: 'noise', label: 'Noise Pattern', icon: 'grain' }
    ];

    const visualizationButtons = visualizations.map(viz => `
        <button 
            onclick="switchVisualization('${viz.id}')" 
            id="viz-${viz.id}"
            class="visualization-btn flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-all ${viz.id === 'original' ? 'bg-primary text-white' : 'bg-card-dark text-slate-400 hover:text-white hover:bg-card-dark/80'}">
            <span class="material-symbols-outlined text-[16px]">${viz.icon}</span>
            <span class="hidden sm:inline">${viz.label}</span>
        </button>
    `).join('');

    return `
        <div class="bg-card-dark border border-[#20324b] rounded-xl p-6">
            <div class="flex items-center justify-between mb-4">
                <h3 class="text-white font-semibold text-lg flex items-center gap-2">
                    <span class="material-symbols-outlined text-primary">image</span>
                    Image Viewer
                </h3>
                <button onclick="openFullscreenViewer()" class="text-slate-400 hover:text-white transition-colors">
                    <span class="material-symbols-outlined">fullscreen</span>
                </button>
            </div>
            
            <!-- Visualization Controls -->
            <div class="flex gap-2 mb-4 overflow-x-auto pb-2">
                ${visualizationButtons}
            </div>
            
            <!-- Image Display -->
            <div class="relative bg-black/20 rounded-lg overflow-hidden">
                <img id="viewerImage" src="${imageData ? imageData.data : ''}" alt="Analysis Image" class="w-full h-auto" />
                ${result.ela && result.ela.ela_image ? `<img id="elaImage" src="data:image/png;base64,${result.ela.ela_image}" class="w-full h-auto hidden" />` : ''}
                ${result.ml_heatmap ? `<img id="heatmapImage" src="data:image/png;base64,${result.ml_heatmap}" class="w-full h-auto hidden" />` : ''}
                ${result.noise_map ? `<img id="noiseImage" src="data:image/png;base64,${result.noise_map}" class="w-full h-auto hidden" />` : ''}
            </div>
            
            <p class="text-slate-400 text-xs mt-3 text-center">
                Click fullscreen icon to view in detail
            </p>
        </div>
    `;
}

/**
 * Build Quick Metrics Grid
 */
function buildQuickMetricsGrid(result) {
    const scores = result.analysis_scores || {};
    const aiProb = result.ai_probability || 0;
    const elaScore = result.ela?.ela_score || 0;

    const metrics = [
        {
            label: 'AI Detection',
            value: `${Math.round(aiProb)}%`,
            detail: 'Overall AI probability',
            icon: 'psychology',
            color: aiProb > 70 ? 'text-red-500' : aiProb > 40 ? 'text-yellow-400' : 'text-accent-green'
        },
        {
            label: 'ELA Score',
            value: elaScore > 0 ? `${Math.round(elaScore)}%` : 'N/A',
            detail: 'Error level analysis',
            icon: 'gradient',
            color: elaScore > 60 ? 'text-red-500' : elaScore > 40 ? 'text-yellow-400' : 'text-accent-green'
        },
        {
            label: 'Metadata Check',
            value: result.metadata?.has_exif ? 'Present' : 'Stripped',
            detail: result.metadata?.ai_tool_detected || 'No AI signatures',
            icon: 'info',
            color: result.metadata?.ai_tool_detected ? 'text-red-500' : 'text-accent-green'
        },
        {
            label: 'Noise Pattern',
            value: `${Math.round(scores.noise_consistency || 0)}%`,
            detail: 'Consistency check',
            icon: 'grain',
            color: (scores.noise_consistency || 0) > 70 ? 'text-red-500' : (scores.noise_consistency || 0) > 50 ? 'text-yellow-400' : 'text-accent-green'
        }
    ];

    return `
        <div class="grid grid-cols-2 lg:grid-cols-4 gap-4">
            ${metrics.map(metric => `
                <div class="bg-card-dark border border-[#20324b] rounded-xl p-4 hover:border-primary/30 transition-colors group">
                    <div class="flex items-start justify-between mb-2">
                        <span class="material-symbols-outlined ${metric.color} text-lg group-hover:scale-110 transition-transform">${metric.icon}</span>
                    </div>
                    <div class="text-2xl font-bold text-white mb-1">${metric.value}</div>
                    <div class="text-[10px] text-slate-400 uppercase tracking-wider mb-1">${metric.label}</div>
                    <div class="text-[10px] text-slate-500">${metric.detail}</div>
                </div>
            `).join('')}
        </div>
    `;
}

/**
 * Build Score Card
 */
function buildScoreCard(result, authenticityScore, verdictConfig) {
    const circumference = 251.2; // 2 * PI * 40
    const offset = circumference - (authenticityScore / 100) * circumference;

    // Custom description for screenshot/watermark/C2PA detection
    let description = result.verdict_description || verdictConfig.description;
    if (result.metadata?.is_screenshot) {
        description = `This is a screen capture, not a camera photo or AI-generated image. Screenshots are considered authentic captures of digital content.`;
    } else if (result.watermark?.watermark_detected) {
        const watermarkType = result.watermark.watermark_type || 'AI watermark';
        description = `AI watermark detected (${watermarkType}). This confirms the image was generated by an AI tool.`;
    } else if (result.content_credentials?.is_ai_generated) {
        const generator = result.content_credentials.ai_generator || 'AI tool';
        description = `Confirmed AI-generated by ${generator} (verified via Content Credentials).`;
    }

    return `
        <div class="bg-card-dark rounded-2xl p-6 border border-white/5 shadow-xl relative overflow-hidden group">
            <div class="absolute top-0 right-0 p-3 opacity-10 group-hover:opacity-20 transition-opacity">
                <span class="material-symbols-outlined text-8xl text-white">shield</span>
            </div>
            
            <h3 class="text-white font-semibold text-lg mb-6 flex items-center gap-2 relative z-10">
                <span class="material-symbols-outlined text-primary">analytics</span>
                Authenticity Score
            </h3>
            
            <div class="flex flex-col items-center justify-center mb-6 relative z-10">
                <div class="relative size-40 flex items-center justify-center">
                    <!-- SVG Gauge Background -->
                    <svg class="transform -rotate-90 size-full" viewbox="0 0 100 100">
                        <circle cx="50" cy="50" fill="none" r="40" stroke="#20324b" stroke-linecap="round"
                            stroke-width="8"></circle>
                        <circle class="transition-all duration-1000 ease-out" cx="50" cy="50" fill="none" r="40"
                            stroke="${verdictConfig.circleColor}" stroke-dasharray="251.2" stroke-dashoffset="${offset}"
                            stroke-linecap="round" stroke-width="8"></circle>
                    </svg>
                    <div class="absolute flex flex-col items-center">
                        <span class="text-4xl font-black text-white">${authenticityScore}</span>
                        <span class="text-xs text-slate-400 font-medium">OUT OF 100</span>
                    </div>
                </div>
                
                <!-- Verdict Badge -->
                <div id="verdictLabel" class="mt-4 px-3 py-1 rounded-full ${verdictConfig.badgeClass} border ${verdictConfig.borderClass} text-sm font-bold tracking-wide">
                    ${verdictConfig.text}
                </div>
            </div>
            
            <!-- Description -->
            <p class="text-slate-400 text-sm text-center leading-relaxed relative z-10">
                ${description}
            </p>
        </div>
    `;
}

/**
 * Build Verification Summary Card
 */
function buildVerificationSummaryCard(result) {
    const aiProb = result.ai_probability || 0;
    const elaScore = result.ela?.ela_score || 0;

    const checks = [
        {
            label: 'AI Detection',
            status: aiProb < 50,
            detail: `${Math.round(aiProb)}% AI likelihood`
        },
        {
            label: 'Watermark Scan',
            status: !result.watermark?.watermark_detected,
            detail: result.watermark?.watermark_detected ? 'AI watermark found' : 'No watermarks detected'
        },
        {
            label: 'Metadata Check',
            status: !result.metadata?.ai_tool_detected,
            detail: result.metadata?.ai_tool_detected || 'No AI tool signatures'
        },
        {
            label: 'ELA Analysis',
            status: elaScore < 60 && !result.ela?.clone_detected,
            detail: result.ela?.clone_detected ? 'Cloning detected' : elaScore > 0 ? `${Math.round(elaScore)}% manipulation` : 'No manipulation found'
        }
    ];

    return `
        <div class="bg-card-dark border border-[#20324b] rounded-xl p-6">
            <h3 class="text-white font-semibold text-lg mb-4 flex items-center gap-2">
                <span class="material-symbols-outlined text-primary">checklist</span>
                Verification Checks
            </h3>
            
            <div class="space-y-3">
                ${checks.map(check => `
                    <div class="flex items-center justify-between p-3 rounded-lg bg-background-dark/50 border border-white/5">
                        <div class="flex items-center gap-3">
                            <div class="size-5 rounded-full ${check.status ? 'bg-accent-green/20' : 'bg-red-500/20'} flex items-center justify-center">
                                <span class="material-symbols-outlined text-xs ${check.status ? 'text-accent-green' : 'text-red-500'}">
                                    ${check.status ? 'check' : 'close'}
                                </span>
                            </div>
                            <div>
                                <div class="text-white text-sm font-medium">${check.label}</div>
                                <div class="text-slate-400 text-xs">${check.detail}</div>
                            </div>
                        </div>
                    </div>
                `).join('')}
            </div>
        </div>
    `;
}

/**
 * Build Quick Actions Card
 */
function buildQuickActionsCard() {
    return `
        <div class="bg-card-dark border border-[#20324b] rounded-xl p-6">
            <h3 class="text-white font-semibold text-lg mb-4 flex items-center gap-2">
                <span class="material-symbols-outlined text-primary">bolt</span>
                Quick Actions
            </h3>
            
            <div class="space-y-3">
                <button onclick="openFeedbackModal()" class="w-full flex items-center gap-3 px-4 py-3 rounded-lg bg-background-dark/50 border border-white/5 text-white hover:border-primary/30 transition-colors text-left">
                    <span class="material-symbols-outlined text-primary">feedback</span>
                    <div>
                        <div class="text-sm font-medium">Report Issue</div>
                        <div class="text-xs text-slate-400">Flag incorrect analysis</div>
                    </div>
                </button>
                
                <button onclick="exportToPDF()" class="w-full flex items-center gap-3 px-4 py-3 rounded-lg bg-background-dark/50 border border-white/5 text-white hover:border-primary/30 transition-colors text-left">
                    <span class="material-symbols-outlined text-primary">download</span>
                    <div>
                        <div class="text-sm font-medium">Export Report</div>
                        <div class="text-xs text-slate-400">Download as PDF</div>
                    </div>
                </button>
                
                <button onclick="handleReanalyze()" class="w-full flex items-center gap-3 px-4 py-3 rounded-lg bg-background-dark/50 border border-white/5 text-white hover:border-primary/30 transition-colors text-left">
                    <span class="material-symbols-outlined text-primary">refresh</span>
                    <div>
                        <div class="text-sm font-medium">Re-analyze</div>
                        <div class="text-xs text-slate-400">Run analysis again</div>
                    </div>
                </button>
            </div>
        </div>
    `;
}

/**
 * Build Key Findings Card
 */
function buildKeyFindingsCard(result) {
    const findings = [];

    // Screenshot detection - show first as it's important context
    if (result.metadata?.is_screenshot) {
        findings.push({
            type: 'info',
            icon: 'screenshot',
            title: 'Screenshot Detected',
            description: 'This is a screen capture, not a camera photo or AI-generated image. Screenshot classification applied.',
            color: 'blue'  // Use blue for screenshot indicators
        });
    }

    // Analyze results and generate findings
    if (result.watermark?.watermark_detected) {
        findings.push({
            type: 'critical',
            icon: 'warning',
            title: 'AI Watermark Detected',
            description: `Found ${result.watermark.watermarks_found.join(', ')} watermark(s) commonly used by AI image generators.`
        });
    }

    if (result.metadata?.ai_tool_detected) {
        findings.push({
            type: 'critical',
            icon: 'warning',
            title: 'AI Tool Signature Found',
            description: `Metadata contains signatures from ${result.metadata.ai_tool_detected}, indicating AI generation.`
        });
    }

    if (result.c2pa?.is_ai_generated) {
        findings.push({
            type: 'critical',
            icon: 'verified_user',
            title: 'Content Credentials (C2PA) Verified',
            description: `Digital signature confirms image was generated by ${result.c2pa.ai_generator || 'an AI tool'}.`
        });
    }

    if (result.ml_prediction?.model === 'FluxDetector' && result.ml_prediction.confidence > 70) {
        findings.push({
            type: 'critical',
            icon: 'psychology',
            title: 'Flux Generation Detected',
            description: `Specialized detector identified characteristics of Flux.1 AI generation (${Math.round(result.ml_prediction.confidence)}% confidence).`
        });
    }

    if (result.ela?.clone_detected) {
        findings.push({
            type: 'warning',
            icon: 'content_copy',
            title: 'Clone Manipulation Detected',
            description: 'Error Level Analysis found evidence of clone stamping or copy-paste manipulation.'
        });
    }

    if ((result.ai_probability || 0) > 80) {
        findings.push({
            type: 'critical',
            icon: 'psychology',
            title: 'High AI Detection Score',
            description: `Analysis shows ${Math.round(result.ai_probability)}% confidence of AI generation.`
        });
    }

    if (result.metadata?.exif_stripped) {
        findings.push({
            type: 'info',
            icon: 'info',
            title: 'EXIF Data Stripped',
            description: 'Camera metadata has been removed, which may indicate post-processing or manipulation.'
        });
    }

    if ((result.analysis_scores?.noise_consistency || 100) < 40) {
        findings.push({
            type: 'warning',
            icon: 'grain',
            title: 'Inconsistent Noise Pattern',
            description: 'Digital noise analysis shows patterns inconsistent with natural camera sensors.'
        });
    }

    // If no significant findings, add a positive note
    if (findings.length === 0) {
        findings.push({
            type: 'success',
            icon: 'check_circle',
            title: 'No Major Issues Detected',
            description: 'Image passed all verification checks with consistent metadata and natural patterns.'
        });
    }

    const typeConfig = {
        critical: { bgClass: 'bg-red-500/10', borderClass: 'border-red-500/20', iconClass: 'text-red-500' },
        warning: { bgClass: 'bg-yellow-400/10', borderClass: 'border-yellow-400/20', iconClass: 'text-yellow-400' },
        info: { bgClass: 'bg-blue-500/10', borderClass: 'border-blue-500/20', iconClass: 'text-blue-500' },
        success: { bgClass: 'bg-accent-green/10', borderClass: 'border-accent-green/20', iconClass: 'text-accent-green' }
    };

    return `
        <div class="bg-card-dark border border-[#20324b] rounded-xl p-6">
            <h3 class="text-white font-semibold text-lg mb-4 flex items-center gap-2">
                <span class="material-symbols-outlined text-primary">lightbulb</span>
                Key Findings
            </h3>
            
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                ${findings.map(finding => {
        const config = typeConfig[finding.type];
        return `
                        <div class="p-4 rounded-lg ${config.bgClass} border ${config.borderClass}">
                            <div class="flex items-start gap-3">
                                <span class="material-symbols-outlined ${config.iconClass} text-xl shrink-0">${finding.icon}</span>
                                <div>
                                    <h4 class="text-white font-medium text-sm mb-1">${finding.title}</h4>
                                    <p class="text-slate-300 text-xs leading-relaxed">${finding.description}</p>
                                </div>
                            </div>
                        </div>
                    `;
    }).join('')}
            </div>
        </div>
    `;
}

// ============================================================================
// Component Builders - Technical Tab
// ============================================================================

/**
 * Build Detection Methods Card
 */
function buildDetectionMethodsCard(result) {
    const aiProb = result.ai_probability || 0;
    const elaScore = result.ela?.ela_score || 0;
    const noiseScore = result.analysis_scores?.noise_consistency || 0;

    const methods = [
        {
            name: 'AI Detection',
            score: aiProb,
            weight: 40,
            color: 'bg-primary',
            description: result.ml_prediction?.model
                ? `ML Analysis (${result.ml_prediction.model}) & Statistics`
                : 'Statistical and ML-based analysis'
        },
        {
            name: 'Error Level Analysis',
            score: elaScore,
            weight: 25,
            color: 'bg-accent-warning',
            description: 'JPEG compression artifact analysis'
        },
        {
            name: 'Metadata Analysis',
            score: result.metadata?.ai_tool_detected ? 100 : 0,
            weight: 20,
            color: 'bg-purple-500',
            description: 'EXIF data and embedded signatures'
        },
        {
            name: 'Noise Analysis',
            score: noiseScore > 0 ? (100 - noiseScore) : 0,
            weight: 15,
            color: 'bg-accent-success',
            description: 'Digital noise pattern consistency'
        }
    ];

    return `
        <div class="bg-card-dark rounded-2xl p-6 border border-white/5 shadow-xl">
            <h3 class="text-white font-semibold text-lg mb-6 flex items-center gap-2">
                <span class="material-symbols-outlined text-primary">science</span>
                Detection Methods
            </h3>
            
            <div class="space-y-5">
                ${methods.map(method => {
        const scoreRounded = Math.round(method.score);
        return `
                        <div class="space-y-2">
                            <div class="flex items-center justify-between">
                                <span class="text-white font-medium text-sm">${method.name}</span>
                                <span class="text-slate-300 text-sm font-semibold">${scoreRounded}% <span class="text-slate-500 font-normal">(${method.weight}% weight)</span></span>
                            </div>
                            <div class="h-3 bg-slate-700/50 rounded-full overflow-hidden">
                                <div class="h-full ${method.color} rounded-full transition-all duration-1000 ease-out" style="width: ${scoreRounded}%"></div>
                            </div>
                            <p class="text-slate-500 text-xs">${method.description}</p>
                        </div>
                    `;
    }).join('')}
            </div>
            
            <div class="mt-6 pt-4 border-t border-white/10">
                <div class="flex items-center justify-between">
                    <span class="text-slate-400 text-sm font-medium">Weighted Combined Score</span>
                    <span class="text-3xl font-black text-primary">${Math.round(result.ai_probability || 50)}%</span>
                </div>
            </div>
        </div>
    `;
}

/**
 * Build Forensic Analysis Card
 */
function buildForensicAnalysisCard(result) {
    const forensics = result.forensics || {};

    return `
        <div class="bg-card-dark border border-[#20324b] rounded-xl p-6">
            <h3 class="text-white font-semibold text-lg mb-4 flex items-center gap-2">
                <span class="material-symbols-outlined text-primary">fingerprint</span>
                Forensic Analysis
            </h3>
            
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div class="p-4 rounded-lg bg-background-dark/50 border border-white/5">
                    <div class="flex items-center gap-2 mb-2">
                        <span class="material-symbols-outlined text-primary">texture</span>
                        <span class="text-white font-medium text-sm">JPEG Quality</span>
                    </div>
                    <div class="text-2xl font-bold text-white mb-1">${result.metadata?.jpeg_quality || 'N/A'}</div>
                    <p class="text-slate-400 text-xs">Compression quality level</p>
                </div>
                
                <div class="p-4 rounded-lg bg-background-dark/50 border border-white/5">
                    <div class="flex items-center gap-2 mb-2">
                        <span class="material-symbols-outlined text-primary">grid_on</span>
                        <span class="text-white font-medium text-sm">Grid Analysis</span>
                    </div>
                    <div class="text-2xl font-bold text-white mb-1">${result.ela?.grid_consistency || 'N/A'}%</div>
                    <p class="text-slate-400 text-xs">8x8 DCT block consistency</p>
                </div>
                
                <div class="p-4 rounded-lg bg-background-dark/50 border border-white/5">
                    <div class="flex items-center gap-2 mb-2">
                        <span class="material-symbols-outlined text-primary">grain</span>
                        <span class="text-white font-medium text-sm">Noise Level</span>
                    </div>
                    <div class="text-2xl font-bold text-white mb-1">${Math.round(result.analysis_scores?.noise_consistency || 0)}%</div>
                    <p class="text-slate-400 text-xs">Digital noise consistency</p>
                </div>
                
                <div class="p-4 rounded-lg bg-background-dark/50 border border-white/5">
                    <div class="flex items-center gap-2 mb-2">
                        <span class="material-symbols-outlined text-primary">contrast</span>
                        <span class="text-white font-medium text-sm">Color Analysis</span>
                    </div>
                    <div class="text-2xl font-bold text-white mb-1">${result.color_anomaly_score || 'N/A'}%</div>
                    <p class="text-slate-400 text-xs">Color space consistency</p>
                </div>
            </div>
        </div>
    `;
}

/**
 * Build ELA Analysis Card
 */
function buildELAAnalysisCard(result) {
    const ela = result.ela || {};

    return `
        <div class="bg-card-dark border border-[#20324b] rounded-xl p-6">
            <h3 class="text-white font-semibold text-lg mb-4 flex items-center gap-2">
                <span class="material-symbols-outlined text-primary">gradient</span>
                Error Level Analysis (ELA)
            </h3>
            
            <p class="text-slate-300 text-sm mb-4">
                ELA reveals areas of an image that are at different compression levels. 
                Bright areas may indicate recent manipulation.
            </p>
            
            <div class="grid grid-cols-2 gap-4 mb-4">
                <div class="p-3 rounded-lg bg-background-dark/50 border border-white/5">
                    <div class="text-slate-400 text-xs mb-1">ELA Score</div>
                    <div class="text-xl font-bold text-white">${Math.round(ela.ela_score || 0)}%</div>
                </div>
                
                <div class="p-3 rounded-lg bg-background-dark/50 border border-white/5">
                    <div class="text-slate-400 text-xs mb-1">Clone Detection</div>
                    <div class="text-xl font-bold ${ela.clone_detected ? 'text-red-500' : 'text-accent-green'}">
                        ${ela.clone_detected ? 'Detected' : 'None'}
                    </div>
                </div>
            </div>
            
            <div class="space-y-3">
                <div class="flex items-start gap-2">
                    <span class="material-symbols-outlined text-primary text-sm mt-0.5">info</span>
                    <p class="text-slate-400 text-xs">
                        <strong class="text-white">Manipulation Likelihood:</strong> ${Math.round(ela.manipulation_likelihood || 0)}%
                    </p>
                </div>
                
                ${ela.high_ela_regions ? `
                    <div class="flex items-start gap-2">
                        <span class="material-symbols-outlined text-yellow-400 text-sm mt-0.5">warning</span>
                        <p class="text-slate-400 text-xs">
                            <strong class="text-white">High ELA Regions:</strong> ${ela.high_ela_regions} areas with suspicious compression levels
                        </p>
                    </div>
                ` : ''}
            </div>
        </div>
    `;
}

/**
 * Build Noise Analysis Card
 */
function buildNoiseAnalysisCard(result) {
    const noiseConsistency = result.analysis_scores?.noise_consistency || 0;

    return `
        <div class="bg-card-dark border border-[#20324b] rounded-xl p-6">
            <h3 class="text-white font-semibold text-lg mb-4 flex items-center gap-2">
                <span class="material-symbols-outlined text-primary">grain</span>
                Noise Pattern Analysis
            </h3>
            
            <p class="text-slate-300 text-sm mb-4">
                Natural camera sensors produce consistent noise patterns. 
                AI-generated images often lack realistic noise or show uniform patterns.
            </p>
            
            <div class="mb-4">
                <div class="flex items-center justify-between text-sm mb-2">
                    <span class="text-white font-medium">Noise Consistency</span>
                    <span class="text-slate-400">${Math.round(noiseConsistency)}%</span>
                </div>
                <div class="h-3 bg-slate-700 rounded-full overflow-hidden">
                    <div class="h-full bg-gradient-to-r from-red-500 via-yellow-400 to-accent-green rounded-full transition-all" style="width: ${noiseConsistency}%"></div>
                </div>
            </div>
            
            <div class="grid grid-cols-3 gap-3">
                <div class="p-3 rounded-lg bg-background-dark/50 border border-white/5 text-center">
                    <div class="text-slate-400 text-xs mb-1">Low Freq</div>
                    <div class="text-lg font-bold text-white">${result.noise_analysis?.low_freq || 'N/A'}</div>
                </div>
                
                <div class="p-3 rounded-lg bg-background-dark/50 border border-white/5 text-center">
                    <div class="text-slate-400 text-xs mb-1">Mid Freq</div>
                    <div class="text-lg font-bold text-white">${result.noise_analysis?.mid_freq || 'N/A'}</div>
                </div>
                
                <div class="p-3 rounded-lg bg-background-dark/50 border border-white/5 text-center">
                    <div class="text-slate-400 text-xs mb-1">High Freq</div>
                    <div class="text-lg font-bold text-white">${result.noise_analysis?.high_freq || 'N/A'}</div>
                </div>
            </div>
            
            <div class="mt-4 p-3 rounded-lg ${noiseConsistency > 70 ? 'bg-accent-green/10 border-accent-green/20' : 'bg-yellow-400/10 border-yellow-400/20'} border">
                <p class="text-xs ${noiseConsistency > 70 ? 'text-accent-green' : 'text-yellow-400'}">
                    ${noiseConsistency > 70
            ? '✓ Noise pattern is consistent with natural camera sensors'
            : '⚠ Noise pattern shows irregularities that may indicate AI generation or heavy processing'}
                </p>
            </div>
        </div>
    `;
}

/**
 * Build Advanced AI Detectors Card
 */
function buildAdvancedDetectorsCard(result) {
    const individualResults = result.individual_results || {};

    // Prepare detector data
    const detectors = [
        {
            id: 'dire',
            name: 'DIRE (Diffusion Recon)',
            icon: 'blur_on',
            result: individualResults.dire,
            description: 'Analyzes diffusion reconstruction error to detect diffusion models (like Stable Diffusion).',
            findings: individualResults.dire?.interpretation
        },
        {
            id: 'npr',
            name: 'NPR (Pixel Relations)',
            icon: 'grid_4x4',
            result: individualResults.npr,
            description: 'Examines neighboring pixel relationships and local anomalies.',
            findings: individualResults.npr?.verdict
        },
        {
            id: 'face',
            name: 'Face Consistency',
            icon: 'face',
            result: individualResults.face_consistency,
            description: 'Checks for facial artifacts like asymmetric eyes and lighting inconsistencies.',
            findings: individualResults.face_consistency?.verdict
        },
        {
            id: 'edge',
            name: 'Edge Coherence',
            icon: 'check_box_outline_blank',
            result: individualResults.edge_coherence,
            description: 'Analyzes edge sharpness and halo artifacts common in AI images.',
            findings: individualResults.edge_coherence?.verdict
        },
        {
            id: 'sbi',
            name: 'SBI Diffusion',
            icon: 'waves',
            result: individualResults.sbi,
            description: 'Self-Blended Images method for detecting latent diffusion artifacts.',
            findings: null
        },
        {
            id: 'forgery',
            name: 'Forgery Detection',
            icon: 'content_cut',
            result: individualResults.forgery,
            description: 'Detects copy-move forgery and cloning within the image.',
            findings: individualResults.forgery?.verdict
        }
    ];

    // Filter to only show detectors that ran
    const activeDetectors = detectors.filter(d => d.result && d.result.success);

    if (activeDetectors.length === 0) return '';

    return `
        <div class="bg-card-dark border border-[#20324b] rounded-xl p-6">
            <h3 class="text-white font-semibold text-lg mb-4 flex items-center gap-2">
                <span class="material-symbols-outlined text-primary">psychology</span>
                Advanced AI Detectors
            </h3>
            
            <p class="text-slate-300 text-sm mb-6">
                Specialized detectors targeting specific AI generation artifacts and manipulation techniques.
            </p>
            
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                ${activeDetectors.map(d => {
        const aiScore = d.result.ai_probability || 0;
        const isAi = aiScore > 50;
        const scoreColor = isAi ? 'text-red-500' : 'text-accent-green';
        const barColor = isAi ? 'bg-red-500' : 'bg-accent-green';
        const bgClass = isAi ? 'bg-red-500/5 border-red-500/10' : 'bg-accent-green/5 border-accent-green/10';

        return `
                        <div class="p-4 rounded-lg border ${bgClass} hover:border-white/10 transition-colors">
                            <div class="flex items-start justify-between mb-3">
                                <div class="flex items-center gap-2">
                                    <span class="material-symbols-outlined text-slate-400">${d.icon}</span>
                                    <h4 class="text-white font-medium text-sm">${d.name}</h4>
                                </div>
                                <div class="flex items-center gap-2">
                                    <span class="text-xs text-slate-400">AI Prob:</span>
                                    <span class="font-bold ${scoreColor}">${Math.round(aiScore)}%</span>
                                </div>
                            </div>
                            
                            <div class="h-1.5 bg-slate-700/50 rounded-full overflow-hidden mb-3">
                                <div class="h-full ${barColor} rounded-full" style="width: ${aiScore}%"></div>
                            </div>
                            
                            <p class="text-slate-400 text-xs leading-relaxed mb-2">
                                ${d.description}
                            </p>
                            
                            ${d.findings ? `
                                <div class="mt-2 pt-2 border-t border-white/5">
                                    <span class="text-xs text-white opacity-80">Finding: ${d.findings}</span>
                                </div>
                            ` : ''}
                            
                            ${d.id === 'face' && d.result.faces_found > 0 ? `
                                <div class="mt-1">
                                    <span class="text-xs text-slate-500">${d.result.faces_found} face(s) analyzed</span>
                                </div>
                            ` : ''}
                        </div>
                    `;
    }).join('')}
            </div>
        </div>
    `;
}

/**
 * Build Watermark Detection Card - Comprehensive Forensic Version
 */
function buildWatermarkDetectionCard(result) {
    const watermark = result.watermark || {};
    const methods = watermark.detection_methods || {};

    return `
        <div class="bg-card-dark border border-[#20324b] rounded-xl p-6">
            <h3 class="text-white font-semibold text-lg mb-4 flex items-center gap-2">
                <span class="material-symbols-outlined text-primary">verified</span>
                Invisible Watermark Detection
            </h3>
            
            <p class="text-slate-300 text-sm mb-4">
                AI generators embed invisible watermarks to identify their output. Advanced forensic analysis using 10 detection methods.
            </p>
            
            <!-- Overall Status Banner -->
            <div class="p-4 rounded-lg ${watermark.watermark_detected ? 'bg-red-500/10 border-red-500/20' : 'bg-accent-green/10 border-accent-green/20'} border mb-6">
                <div class="flex items-center gap-3 mb-2">
                    <span class="text-3xl">${watermark.watermark_detected ? '⚠️' : '✅'}</span>
                    <div class="flex-1">
                        <div class="text-white font-semibold text-lg">
                            ${watermark.watermark_detected ? 'Watermark Detected' : 'No Watermark Found'}
                        </div>
                        ${watermark.watermark_detected ? `
                            <div class="text-sm text-slate-300 mt-1">
                                Primary Type: <span class="font-medium text-white">${escapeHTML(watermark.watermark_type || 'Unknown')}</span> 
                                <span class="ml-2 px-2 py-0.5 rounded bg-red-500/20 text-red-400 text-xs">${watermark.confidence}% confidence</span>
                            </div>
                        ` : `
                            <div class="text-sm text-slate-400 mt-1">
                                Scanned with 10 detection methods - No invisible watermarks found
                            </div>
                        `}
                    </div>
                </div>
                
                ${watermark.ai_generator_signature ? `
                    <div class="mt-3 p-2 rounded bg-background-dark/50 border border-white/10">
                        <span class="text-xs text-slate-400">🎯 Identified Generator:</span>
                        <span class="text-white font-medium ml-2">${formatGeneratorName(watermark.ai_generator_signature)}</span>
                    </div>
                ` : ''}
            </div>
            
            <!-- Detection Methods Breakdown -->
            <div class="mb-6">
                <div class="flex items-center justify-between mb-3">
                    <h4 class="text-white font-medium text-sm">Detection Methods Analysis</h4>
                    <button onclick="toggleWatermarkFilters()" class="text-xs text-primary hover:text-blue-400 transition-colors">
                        Filter Methods
                    </button>
                </div>
                
                <div class="space-y-2" id="watermarkMethodsList">
                    ${buildWatermarkMethodRow('DWT-DCT Watermark', methods.invisible_watermark, 'Stable Diffusion, ComfyUI, AUTOMATIC1111')}
                    ${buildWatermarkMethodRow('Meta Stable Signature', methods.stable_signature, 'Meta AI generators - 48-bit neural watermark')}
                    ${buildWatermarkMethodRow('Spectral Analysis', methods.spectral_analysis, 'Frequency-domain patterns, Tree-Ring detection')}
                    ${buildWatermarkMethodRow('LSB Steganography', methods.lsb_analysis, 'Least Significant Bit watermarks')}
                    ${buildWatermarkMethodRow('Tree-Ring Pattern', methods.treering_analysis, 'Academic diffusion watermarking method')}
                    ${buildWatermarkMethodRow('Gaussian Shading', methods.gaussian_shading, 'Variance-based watermark detection')}
                    ${buildWatermarkMethodRow('Metadata Markers', methods.metadata_watermark, 'IPTC DigitalSourceType, EXIF AI markers')}
                    ${buildWatermarkMethodRow('SteganoGAN', methods.steganogan, 'GAN-based steganographic watermarks')}
                    ${buildWatermarkMethodRow('Adversarial (Exp.)', methods.adversarial_analysis, 'Glaze/Nightshade detection - experimental')}
                </div>
            </div>
            
            <!-- Detailed Findings -->
            ${watermark.details && watermark.details.length > 0 ? `
                <div class="border-t border-white/10 pt-4 mb-4">
                    <h4 class="text-white font-medium text-sm mb-3 flex items-center gap-2">
                        <span class="material-symbols-outlined text-primary text-base">description</span>
                        Detailed Findings
                    </h4>
                    <ul class="space-y-2">
                        ${watermark.details.map(detail => `
                            <li class="flex items-start gap-2 text-sm text-slate-300">
                                <span class="text-primary mt-0.5">•</span>
                                <span>${escapeHTML(detail)}</span>
                            </li>
                        `).join('')}
                    </ul>
                </div>
            ` : ''}
            
            <!-- Watermark Visualization (if heatmap available) -->
            ${watermark.visualizations?.watermark_heatmap ? `
                <div class="border-t border-white/10 pt-4">
                    <h4 class="text-white font-medium text-sm mb-3 flex items-center gap-2">
                        <span class="material-symbols-outlined text-primary text-base">heat_pump</span>
                        Spatial Watermark Heatmap
                    </h4>
                    <div class="relative rounded-lg overflow-hidden border border-white/10">
                        <img src="${watermark.visualizations.watermark_heatmap}" 
                             alt="Watermark Heatmap" 
                             class="w-full h-auto"
                             style="image-rendering: pixelated;">
                        <div class="absolute bottom-2 right-2 px-2 py-1 rounded bg-black/60 text-xs text-white">
                            Red = High watermark strength
                        </div>
                    </div>
                    <p class="text-xs text-slate-400 mt-2">
                        Heatmap showing spatial distribution of watermark signals across the image
                    </p>
                </div>
            ` : ''}
            
            <!-- Known Limitations Note -->
            ${methods.synthid ? `
                <div class="mt-4 p-3 rounded-lg bg-blue-500/10 border border-blue-500/20">
                    <div class="flex items-start gap-2">
                        <span class="material-symbols-outlined text-blue-400 text-sm mt-0.5">info</span>
                        <p class="text-xs text-blue-300">
                            <strong>Note:</strong> Google SynthID watermarks cannot be detected externally (requires Google API). 
                            This analysis covers DWT-DCT, Stable Signature, and other open watermarking standards.
                        </p>
                    </div>
                </div>
            ` : ''}
        </div>
    `;
}

/**
 * Build individual watermark method detection row
 */
function buildWatermarkMethodRow(name, methodResult, description) {
    if (!methodResult) {
        return `
            <div class="flex items-center justify-between p-2 rounded bg-background-dark/30 border border-white/5 opacity-50">
                <div class="flex items-center gap-2">
                    <span class="material-symbols-outlined text-slate-500 text-sm">cancel</span>
                    <div>
                        <span class="text-sm text-slate-400">${escapeHTML(name)}</span>
                        ${description ? `<div class="text-xs text-slate-500 mt-0.5">${escapeHTML(description)}</div>` : ''}
                    </div>
                </div>
                <span class="text-xs text-slate-500">Not run</span>
            </div>
        `;
    }

    const detected = methodResult.detected || methodResult.patterns_found || methodResult.anomaly_detected || false;
    const confidence = methodResult.confidence || 0;
    const hasError = methodResult.error;
    const hasNote = methodResult.note;

    return `
        <div class="flex items-center justify-between p-3 rounded ${detected ? 'bg-red-500/10 border-red-500/20' : 'bg-background-dark/30 border-white/5'} border transition-all hover:bg-background-dark/50">
            <div class="flex items-center gap-3 flex-1">
                <span class="material-symbols-outlined text-sm ${detected ? 'text-red-500' : hasError ? 'text-yellow-500' : 'text-slate-500'}">
                    ${detected ? 'check_circle' : hasError ? 'error' : 'cancel'}
                </span>
                <div class="flex-1">
                    <div class="flex items-center gap-2">
                        <span class="text-sm ${detected ? 'text-white font-medium' : 'text-slate-400'}">${escapeHTML(name)}</span>
                        ${hasNote ? `<span class="text-xs text-blue-400" title="${escapeHTML(hasNote)}">ℹ️</span>` : ''}
                    </div>
                    ${description ? `
                        <div class="text-xs text-slate-500 mt-0.5">${escapeHTML(description)}</div>
                    ` : ''}
                    ${detected && methodResult.type ? `
                        <div class="text-xs text-slate-400 mt-1">Type: <span class="text-white">${escapeHTML(methodResult.type)}</span></div>
                    ` : ''}
                </div>
            </div>
            
            <div class="flex items-center gap-3">
                ${detected ? `
                    <div class="flex items-center gap-2">
                        <div class="w-24 h-2 bg-slate-700 rounded-full overflow-hidden">
                            <div class="h-full bg-gradient-to-r from-red-500 to-red-400 rounded-full transition-all" 
                                 style="width: ${confidence}%"></div>
                        </div>
                        <div class="px-2 py-0.5 rounded text-xs font-medium ${confidence > 70 ? 'bg-red-500/20 text-red-400' : 'bg-yellow-500/20 text-yellow-400'}">
                            ${confidence}%
                        </div>
                    </div>
                ` : hasError ? `
                    <span class="text-xs text-yellow-400" title="${escapeHTML(methodResult.error)}">⚠️ Error</span>
                ` : `
                    <span class="text-xs text-slate-500">Not detected</span>
                `}
            </div>
        </div>
    `;
}

/**
 * Format generator name for display
 */
function formatGeneratorName(signature) {
    const nameMap = {
        'stable_diffusion': 'Stable Diffusion',
        'sd_webui': 'AUTOMATIC1111 WebUI',
        'comfyui': 'ComfyUI',
        'invokeai': 'InvokeAI',
        'sdxl': 'Stable Diffusion XL',
        'sd_turbo': 'SD Turbo',
        'leonardo': 'Leonardo AI',
        'runwayml': 'Runway ML',
        'playground': 'Playground AI',
        'ideogram': 'Ideogram',
        'flux': 'FLUX',
        'juggernaut': 'Juggernaut',
        'imagen': 'Google Imagen',
        'parti': 'Google Parti',
        'muse': 'Google Muse',
        'meta_ai': 'Meta AI',
        'trainedAlgorithmicMedia': 'AI Generated (IPTC)',
        'compositeWithTrainedAlgorithmicMedia': 'AI Composite (IPTC)',
        'algorithmicMedia': 'Algorithmic Media'
    };

    return nameMap[signature] || signature.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}

/**
 * Toggle watermark method filters (placeholder for future implementation)
 */
function toggleWatermarkFilters() {
    // Future: Add filter UI (All/Detected Only/High Confidence)
    console.log('Filter toggle - feature to be implemented');
}

// ============================================================================
// Component Builders - Metadata Tab
// ============================================================================

/**
 * Build Metadata Overview Card
 */
function buildMetadataOverviewCard(result) {
    const metadata = result.metadata || {};

    return `
        <div class="bg-card-dark border border-[#20324b] rounded-xl p-6">
            <h3 class="text-white font-semibold text-lg mb-4 flex items-center gap-2">
                <span class="material-symbols-outlined text-primary">summarize</span>
                Metadata Overview
            </h3>
            
            <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div class="p-4 rounded-lg bg-background-dark/50 border border-white/5">
                    <div class="text-slate-400 text-xs mb-2">EXIF Present</div>
                    <div class="text-xl font-bold ${metadata.has_exif ? 'text-accent-green' : 'text-red-500'}">
                        ${metadata.has_exif ? 'Yes' : 'No'}
                    </div>
                </div>
                
                <div class="p-4 rounded-lg bg-background-dark/50 border border-white/5">
                    <div class="text-slate-400 text-xs mb-2">Image Type</div>
                    <div class="text-xl font-bold ${metadata.is_screenshot ? 'text-blue-400' : 'text-white'}">
                        ${metadata.is_screenshot ? 'Screenshot' : 'Photo'}
                    </div>
                </div>
                
                <div class="p-4 rounded-lg bg-background-dark/50 border border-white/5">
                    <div class="text-slate-400 text-xs mb-2">AI Tool</div>
                    <div class="text-xl font-bold ${metadata.ai_tool_detected ? 'text-red-500' : 'text-accent-green'}">
                        ${metadata.ai_tool_detected || 'None'}
                    </div>
                </div>
                
                <div class="p-4 rounded-lg bg-background-dark/50 border border-white/5">
                    <div class="text-slate-400 text-xs mb-2">Modified</div>
                    <div class="text-xl font-bold ${metadata.exif_stripped ? 'text-yellow-400' : 'text-white'}">
                        ${metadata.exif_stripped ? 'Yes' : 'No'}
                    </div>
                </div>
            </div>
            
            ${metadata.is_screenshot ? `
                <div class="mt-4 p-4 rounded-lg bg-blue-500/10 border border-blue-500/20">
                    <div class="flex items-start gap-3">
                        <span class="material-symbols-outlined text-blue-400">screenshot</span>
                        <div class="flex-1">
                            <div class="text-white font-medium mb-1">Screenshot Detected</div>
                            <p class="text-slate-300 text-sm mb-2">
                                This appears to be a screen capture, not a camera photo or AI-generated image.
                            </p>
                            ${metadata.screenshot_indicators && metadata.screenshot_indicators.length > 0 ? `
                                <div class="text-xs text-slate-400">
                                    <strong>Indicators:</strong>
                                    <ul class="list-disc list-inside mt-1">
                                        ${metadata.screenshot_indicators.map(ind => `<li>${escapeHTML(ind)}</li>`).join('')}
                                    </ul>
                                </div>
                            ` : ''}
                        </div>
                    </div>
                </div>
            ` : ''}
            
            ${metadata.ai_tool_detected ? `
                <div class="mt-4 p-4 rounded-lg bg-red-500/10 border border-red-500/20">
                    <div class="flex items-start gap-3">
                        <span class="material-symbols-outlined text-red-500">warning</span>
                        <div>
                            <div class="text-white font-medium mb-1">AI Generation Tool Detected</div>
                            <p class="text-slate-300 text-sm">
                                Metadata contains signatures from <strong>${metadata.ai_tool_detected}</strong>, 
                                strongly indicating this is an AI-generated image.
                            </p>
                        </div>
                    </div>
                </div>
            ` : ''}
        </div>
    `;
}

/**
 * Build Watermark Metadata Card
 */
function buildWatermarkMetadataCard(result) {
    const watermark = result.watermark || {};
    const metadataWm = watermark.detection_methods?.metadata_watermark || {};

    // Only show if metadata watermark was found
    if (!metadataWm.found && !watermark.ai_generator_signature) {
        return ''; // Don't show card if no metadata-based watermarks
    }

    return `
        <div class="bg-card-dark border border-[#20324b] rounded-xl p-6">
            <h3 class="text-white font-semibold text-lg mb-4 flex items-center gap-2">
                <span class="material-symbols-outlined text-primary">verified_user</span>
                Watermark Metadata
            </h3>
            
            <div class="p-4 rounded-lg bg-red-500/10 border border-red-500/20">
                <div class="flex items-start gap-3">
                    <span class="material-symbols-outlined text-red-500 text-2xl">warning</span>
                    <div class="flex-1">
                        <div class="text-white font-medium mb-2">AI Watermark Found in Metadata</div>
                        
                        ${metadataWm.type ? `
                            <div class="mb-2">
                                <span class="text-slate-400 text-sm">Watermark Type:</span>
                                <span class="text-white font-medium ml-2">${escapeHTML(metadataWm.type)}</span>
                            </div>
                        ` : ''}
                        
                        ${metadataWm.ai_source_type ? `
                            <div class="mb-2">
                                <span class="text-slate-400 text-sm">IPTC Source Type:</span>
                                <span class="text-white font-medium ml-2">${escapeHTML(metadataWm.ai_source_type)}</span>
                            </div>
                        ` : ''}
                        
                        ${watermark.ai_generator_signature ? `
                            <div class="mb-2">
                                <span class="text-slate-400 text-sm">Generator Identified:</span>
                                <span class="text-white font-medium ml-2">${formatGeneratorName(watermark.ai_generator_signature)}</span>
                            </div>
                        ` : ''}
                        
                        ${metadataWm.content ? `
                            <div class="mt-3 p-2 rounded bg-background-dark/50 border border-white/10">
                                <div class="text-slate-400 text-xs mb-1">Watermark Content:</div>
                                <div class="text-white text-sm font-mono break-all">${escapeHTML(metadataWm.content)}</div>
                            </div>
                        ` : ''}
                        
                        <p class="text-slate-300 text-sm mt-3">
                            Metadata-based watermarks are embedded in image file headers (EXIF, IPTC, XMP). 
                            They indicate AI generation with high reliability.
                        </p>
                    </div>
                </div>
            </div>
            
            ${metadataWm.fields_found && metadataWm.fields_found.length > 0 ? `
                <div class="mt-4">
                    <div class="text-slate-400 text-xs uppercase tracking-wider mb-2">Fields Checked:</div>
                    <div class="flex flex-wrap gap-2">
                        ${metadataWm.fields_found.map(field => `
                            <span class="px-2 py-1 rounded bg-background-dark/50 border border-white/10 text-xs text-slate-300">
                                ${escapeHTML(field)}
                            </span>
                        `).join('')}
                    </div>
                </div>
            ` : ''}
        </div>
    `;
}

/**
 * Build EXIF Data Card
 */
function buildEXIFDataCard(result) {
    const exif = result.metadata?.exif_data || {};
    const hasData = Object.keys(exif).length > 0;

    return `
        <div class="bg-card-dark border border-[#20324b] rounded-xl p-6">
            <h3 class="text-white font-semibold text-lg mb-4 flex items-center gap-2">
                <span class="material-symbols-outlined text-primary">data_object</span>
                EXIF Data
            </h3>
            
            ${hasData ? `
                <div class="space-y-2 max-h-96 overflow-y-auto custom-scrollbar">
                    ${Object.entries(exif).map(([key, value]) => `
                        <div class="flex items-start justify-between p-3 rounded-lg bg-background-dark/50 border border-white/5 hover:border-primary/30 transition-colors">
                            <span class="text-slate-400 text-sm">${escapeHTML(key)}</span>
                            <span class="text-white text-sm font-mono text-right ml-4">${escapeHTML(String(value))}</span>
                        </div>
                    `).join('')}
                </div>
            ` : `
                <div class="text-center py-8">
                    <span class="material-symbols-outlined text-slate-500 text-5xl mb-3 block">hide_source</span>
                    <p class="text-slate-400">No EXIF data found</p>
                    <p class="text-slate-500 text-sm mt-2">
                        EXIF data may have been stripped during processing or the image was generated without camera metadata.
                    </p>
                </div>
            `}
        </div>
    `;
}

/**
 * Build C2PA Credentials Card
 */
function buildC2PACredentialsCard(result) {
    const c2pa = result.content_credentials || {};

    return `
        <div class="bg-card-dark border border-[#20324b] rounded-xl p-6">
            <h3 class="text-white font-semibold text-lg mb-4 flex items-center gap-2">
                <span class="material-symbols-outlined text-primary">verified_user</span>
                Content Credentials (C2PA)
            </h3>
            
            <p class="text-slate-300 text-sm mb-4">
                C2PA is a standard for content authenticity and provenance. Presence of C2PA data indicates 
                the image has a verified chain of custody.
            </p>
            
            <div class="p-4 rounded-lg ${c2pa.c2pa_found ? 'bg-accent-green/10 border-accent-green/20' : 'bg-slate-500/10 border-slate-500/20'} border">
                <div class="flex items-center gap-3 mb-3">
                    <span class="material-symbols-outlined ${c2pa.c2pa_found ? 'text-accent-green' : 'text-slate-400'} text-2xl">
                        ${c2pa.c2pa_found ? 'verified_user' : 'shield'}
                    </span>
                    <div>
                        <div class="text-white font-medium">
                            ${c2pa.c2pa_found ? 'C2PA Manifest Found' : 'No C2PA Data'}
                        </div>
                        <div class="text-slate-300 text-xs mt-1">
                            ${c2pa.c2pa_found
            ? 'Image contains verified content credentials'
            : 'No content provenance information available'}
                        </div>
                    </div>
                </div>
                
                ${c2pa.c2pa_found && c2pa.manifest ? `
                    <div class="mt-4 space-y-2">
                        ${c2pa.manifest.creator ? `
                            <div class="flex items-center gap-2 text-sm">
                                <span class="text-slate-400">Creator:</span>
                                <span class="text-white">${escapeHTML(c2pa.manifest.creator)}</span>
                            </div>
                        ` : ''}
                        ${c2pa.manifest.claim_generator ? `
                            <div class="flex items-center gap-2 text-sm">
                                <span class="text-slate-400">Tool:</span>
                                <span class="text-white">${escapeHTML(c2pa.manifest.claim_generator)}</span>
                            </div>
                        ` : ''}
                        ${c2pa.manifest.timestamp ? `
                            <div class="flex items-center gap-2 text-sm">
                                <span class="text-slate-400">Timestamp:</span>
                                <span class="text-white">${escapeHTML(c2pa.manifest.timestamp)}</span>
                            </div>
                        ` : ''}
                    </div>
                ` : ''}
            </div>
        </div>
    `;
}

/**
 * Build File Properties Card
 */
function buildFilePropertiesCard(result) {
    // Format file size
    const formatFileSize = (bytes) => {
        if (!bytes || bytes === 'Unknown') return 'Unknown';
        if (bytes < 1024) return `${bytes} B`;
        if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
        return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
    };

    return `
        <div class="bg-card-dark border border-[#20324b] rounded-xl p-6">
            <h3 class="text-white font-semibold text-lg mb-4 flex items-center gap-2">
                <span class="material-symbols-outlined text-primary">description</span>
                File Properties
            </h3>
            
            <div class="grid grid-cols-2 gap-4">
                <div class="p-3 rounded-lg bg-background-dark/50 border border-white/5">
                    <div class="text-slate-400 text-xs mb-1">File Name</div>
                    <div class="text-white text-sm font-mono truncate">${imageData?.fileName || 'Unknown'}</div>
                </div>
                
                <div class="p-3 rounded-lg bg-background-dark/50 border border-white/5">
                    <div class="text-slate-400 text-xs mb-1">File Type</div>
                    <div class="text-white text-sm">${imageData?.mimeType || 'Unknown'}</div>
                </div>
                
                <div class="p-3 rounded-lg bg-background-dark/50 border border-white/5">
                    <div class="text-slate-400 text-xs mb-1">Dimensions</div>
                    <div class="text-white text-sm">${result.dimensions?.width || '?'} x ${result.dimensions?.height || '?'}</div>
                </div>
                
                <div class="p-3 rounded-lg bg-background-dark/50 border border-white/5">
                    <div class="text-slate-400 text-xs mb-1">File Size</div>
                    <div class="text-white text-sm">${formatFileSize(result.file_size)}</div>
                </div>
                
                <div class="p-3 rounded-lg bg-background-dark/50 border border-white/5">
                    <div class="text-slate-400 text-xs mb-1">Color Space</div>
                    <div class="text-white text-sm">${result.color_space || 'RGB'}</div>
                </div>
                
                <div class="p-3 rounded-lg bg-background-dark/50 border border-white/5">
                    <div class="text-slate-400 text-xs mb-1">Bit Depth</div>
                    <div class="text-white text-sm">${result.bit_depth ? `${result.bit_depth}-bit` : '24-bit'}</div>
                </div>
            </div>
        </div>
    `;
}

// ============================================================================
// Component Builders - AI Analysis Tab
// ============================================================================

/**
 * Build AI Explanation Card
 */
function buildAIExplanationCard(result) {
    const aiAnalysis = result.ai_analysis || {};

    return `
        <div class="bg-card-dark border border-[#20324b] rounded-xl p-6">
            <h3 class="text-white font-semibold text-lg mb-4 flex items-center gap-2">
                <span class="material-symbols-outlined text-primary">psychology</span>
                AI Visual Analysis
            </h3>
            
            ${aiAnalysis.explanation ? `
                <div class="prose prose-invert max-w-none">
                    <p class="text-slate-300 text-sm leading-relaxed whitespace-pre-line">${escapeHTML(aiAnalysis.explanation)}</p>
                </div>
            ` : ''}

            ${result.ml_prediction?.model ? `
                <div class="mt-4 p-4 rounded-lg bg-background-dark/50 border border-white/5">
                    <div class="flex items-start gap-3">
                        <span class="material-symbols-outlined text-primary">model_training</span>
                        <div>
                            <div class="text-white font-medium text-sm">Model Triggered: ${result.ml_prediction.model}</div>
                            <div class="text-slate-400 text-xs mt-1">
                                ${result.ml_prediction.specialization || 'General AI Detection Model'}
                            </div>
                            <div class="mt-2 flex items-center gap-2">
                                <div class="w-24 h-1.5 bg-slate-700 rounded-full overflow-hidden">
                                    <div class="h-full bg-primary rounded-full" style="width: ${result.ml_prediction.confidence}%"></div>
                                </div>
                                <span class="text-xs text-primary font-bold">${Math.round(result.ml_prediction.confidence)}% Confidence</span>
                            </div>
                        </div>
                    </div>
                </div>
            ` : result.ai_analysis?.explanation ? '' : `
                <p class="text-slate-400 text-sm">AI analysis not available for this image.</p>
            `}
            
            ${aiAnalysis.artifacts_detected && aiAnalysis.artifacts_detected.length > 0 ? `
                <div class="mt-6">
                    <h4 class="text-white font-medium text-sm mb-3 flex items-center gap-2">
                        <span class="material-symbols-outlined text-yellow-400">warning</span>
                        Detected Artifacts
                    </h4>
                    <div class="space-y-2">
                        ${aiAnalysis.artifacts_detected.map(artifact => `
                            <div class="p-3 rounded-lg bg-yellow-400/10 border border-yellow-400/20">
                                <div class="text-yellow-400 text-sm font-medium mb-1">${escapeHTML(artifact.type || 'Unknown')}</div>
                                <div class="text-slate-300 text-xs">${escapeHTML(artifact.description || '')}</div>
                            </div>
                        `).join('')}
                    </div>
                </div>
            ` : ''}
        </div>
    `;
}

/**
 * Build Visual Analysis Card
 */
function buildVisualAnalysisCard(result) {
    const aiAnalysis = result.ai_analysis || {};

    return `
        <div class="bg-card-dark border border-[#20324b] rounded-xl p-6">
            <h3 class="text-white font-semibold text-lg mb-4 flex items-center gap-2">
                <span class="material-symbols-outlined text-primary">visibility</span>
                Visual Inspection Results
            </h3>
            
            <div class="space-y-4">
                ${aiAnalysis.composition_analysis ? `
                    <div>
                        <h4 class="text-white font-medium text-sm mb-2">Composition</h4>
                        <p class="text-slate-300 text-sm">${escapeHTML(aiAnalysis.composition_analysis)}</p>
                    </div>
                ` : ''}
                
                ${aiAnalysis.lighting_analysis ? `
                    <div>
                        <h4 class="text-white font-medium text-sm mb-2">Lighting & Shadows</h4>
                        <p class="text-slate-300 text-sm">${escapeHTML(aiAnalysis.lighting_analysis)}</p>
                    </div>
                ` : ''}
                
                ${aiAnalysis.texture_analysis ? `
                    <div>
                        <h4 class="text-white font-medium text-sm mb-2">Texture & Details</h4>
                        <p class="text-slate-300 text-sm">${escapeHTML(aiAnalysis.texture_analysis)}</p>
                    </div>
                ` : ''}
                
                ${!aiAnalysis.composition_analysis && !aiAnalysis.lighting_analysis && !aiAnalysis.texture_analysis ? `
                    <p class="text-slate-400 text-sm text-center py-4">
                        Detailed visual analysis not available. Enable AI analysis in settings.
                    </p>
                ` : ''}
            </div>
        </div>
    `;
}

/**
 * Build Confidence Breakdown Card
 */
function buildConfidenceBreakdownCard(result) {
    const scores = result.analysis_scores || {};
    const aiProb = result.ai_probability || 0;
    const elaScore = result.ela?.ela_score || 0;
    const noiseScore = scores.noise_consistency || 0;

    const breakdown = [
        { label: 'AI Detection', value: aiProb, max: 100, weight: 40 },
        { label: 'ELA Analysis', value: elaScore, max: 100, weight: 25 },
        { label: 'Metadata Check', value: result.metadata?.ai_tool_detected ? 100 : 0, max: 100, weight: 20 },
        { label: 'Noise Analysis', value: noiseScore > 0 ? (100 - noiseScore) : 0, max: 100, weight: 15 }
    ];

    return `
        <div class="bg-card-dark border border-[#20324b] rounded-xl p-6">
            <h3 class="text-white font-semibold text-lg mb-4 flex items-center gap-2">
                <span class="material-symbols-outlined text-primary">analytics</span>
                Confidence Breakdown
            </h3>
            
            <p class="text-slate-300 text-sm mb-6">
                Our detection system combines multiple analysis methods with different weights 
                to produce the final AI probability score.
            </p>
            
            <div class="space-y-4">
                ${breakdown.map(item => {
        const contribution = (item.value / item.max) * item.weight;
        return `
                        <div class="space-y-2">
                            <div class="flex items-center justify-between text-sm">
                                <span class="text-white">${item.label}</span>
                                <div class="flex items-center gap-2">
                                    <span class="text-slate-400">${Math.round(item.value)}%</span>
                                    <span class="text-slate-500">×</span>
                                    <span class="text-primary">${item.weight}%</span>
                                    <span class="text-slate-500">=</span>
                                    <span class="text-white font-medium">${contribution.toFixed(1)}</span>
                                </div>
                            </div>
                            <div class="h-2 bg-slate-700 rounded-full overflow-hidden">
                                <div class="h-full bg-gradient-to-r from-primary to-purple-500 rounded-full transition-all" style="width: ${item.value}%"></div>
                            </div>
                        </div>
                    `;
    }).join('')}
            </div>
            
            <div class="mt-6 pt-4 border-t border-white/5">
                <div class="flex items-center justify-between">
                    <span class="text-white font-medium">Final AI Probability</span>
                    <span class="text-3xl font-black text-primary">${Math.round(result.ai_probability || 50)}%</span>
                </div>
                <p class="text-slate-400 text-xs mt-2">
                    Based on weighted combination of all detection methods
                </p>
            </div>
        </div>
    `;
}

// ============================================================================
// Visualization Switching
// ============================================================================

/**
 * Switch between different image visualizations
 */
function switchVisualization(vizType) {
    currentVisualization = vizType;

    // Update button states
    document.querySelectorAll('.visualization-btn').forEach(btn => {
        const btnType = btn.id.replace('viz-', '');
        if (btnType === vizType) {
            btn.className = 'visualization-btn flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-all bg-primary text-white';
        } else {
            btn.className = 'visualization-btn flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-all bg-card-dark text-slate-400 hover:text-white hover:bg-card-dark/80';
        }
    });

    // Show/hide images
    const images = {
        'original': document.getElementById('viewerImage'),
        'ela': document.getElementById('elaImage'),
        'heatmap': document.getElementById('heatmapImage'),
        'noise': document.getElementById('noiseImage')
    };

    Object.entries(images).forEach(([type, img]) => {
        if (img) {
            if (type === vizType) {
                img.classList.remove('hidden');
            } else {
                img.classList.add('hidden');
            }
        }
    });
}

// ============================================================================
// Modal Functions
// ============================================================================

/**
 * Open fullscreen image viewer
 */
function openFullscreenViewer() {
    const modal = document.getElementById('fullscreenModal');
    const fullscreenImage = document.getElementById('fullscreenImage');

    if (modal && fullscreenImage && imageData) {
        // Set the appropriate image based on current visualization
        let imgSrc = imageData.data;

        if (currentVisualization === 'ela' && currentResult?.ela?.ela_image) {
            imgSrc = 'data:image/png;base64,' + currentResult.ela.ela_image;
        } else if (currentVisualization === 'heatmap' && currentResult?.ml_heatmap) {
            imgSrc = 'data:image/png;base64,' + currentResult.ml_heatmap;
        } else if (currentVisualization === 'noise' && currentResult?.noise_map) {
            imgSrc = 'data:image/png;base64,' + currentResult.noise_map;
        }

        fullscreenImage.src = imgSrc;
        modal.classList.remove('hidden');
        modal.classList.add('flex');
    }
}

/**
 * Close fullscreen viewer
 */
function closeFullscreenViewer() {
    const modal = document.getElementById('fullscreenModal');
    if (modal) {
        modal.classList.add('hidden');
        modal.classList.remove('flex');
    }
}

/**
 * Open feedback modal
 */
function openFeedbackModal() {
    const modal = document.getElementById('feedbackModal');
    if (modal) {
        modal.classList.remove('hidden');
        modal.classList.add('flex');
    }
}

/**
 * Close feedback modal
 */
function closeFeedbackModal() {
    const modal = document.getElementById('feedbackModal');
    if (modal) {
        modal.classList.add('hidden');
        modal.classList.remove('flex');
    }
}

/**
 * Submit feedback
 */
function submitFeedback() {
    const feedbackType = document.getElementById('feedbackType')?.value;
    const feedbackDescription = document.getElementById('feedbackDescription')?.value;

    if (!feedbackDescription || feedbackDescription.trim() === '') {
        UI.showNotification('Please provide a description of the issue.', 'warning');
        return;
    }

    // Here you would send feedback to backend
    console.log('[Feedback]', { type: feedbackType, description: feedbackDescription, result: currentResult });

    // Show success message
    UI.showNotification('Thank you for your feedback! It will help us improve our detection accuracy.', 'success');

    // Close modal and reset form
    closeFeedbackModal();
    if (document.getElementById('feedbackDescription')) {
        document.getElementById('feedbackDescription').value = '';
    }
}

// ============================================================================
// Event Listeners & Re-attachment
// ============================================================================

/**
 * Attach event listeners to dynamically created elements
 */
function attachEventListeners() {
    // Visualization buttons are handled via onclick in HTML
    // Other dynamic elements can be attached here
}

// ============================================================================
// Export & Actions
// ============================================================================

/**
 * Export analysis report to PDF
 */
function exportToPDF() {
    if (!currentResult) {
        UI.showNotification('No analysis result to export', 'warning');
        return;
    }

    // Simple PDF export - could be enhanced with a library like jsPDF
    UI.showNotification('PDF export functionality would generate a comprehensive report with all analysis details, scores, and visualizations.', 'info');

    // Placeholder for actual PDF generation
    console.log('[Export] Would generate PDF with:', currentResult);
}

/**
 * Handle re-analyze button click
 */
async function handleReanalyze() {
    if (!imageData) {
        UI.showNotification('No image data available to re-analyze', 'error');
        return;
    }

    const confirmReanalyze = confirm('Re-run the analysis on this image?');
    if (confirmReanalyze) {
        await processImageAnalysis(imageData);
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Get verdict configuration for styling
 */
function getVerdictConfig(verdict) {
    const configs = {
        'AI_GENERATED': {
            text: 'AI GENERATED',
            circleColor: 'rgba(239, 68, 68, 1)',
            glowClass: 'bg-red-500/10',
            badgeClass: 'bg-red-500/20 text-red-400',
            borderClass: 'border-red-500/30',
            description: 'This image shows strong indicators of AI generation including synthetic patterns and potential watermarks.'
        },
        'LIKELY_AI': {
            text: 'LIKELY AI GENERATED',
            circleColor: 'rgba(251, 146, 60, 1)',
            glowClass: 'bg-orange-500/10',
            badgeClass: 'bg-orange-500/20 text-orange-400',
            borderClass: 'border-orange-500/30',
            description: 'This image shows characteristics consistent with AI generation. Further manual review recommended.'
        },
        'UNCERTAIN': {
            text: 'UNCERTAIN',
            circleColor: 'rgba(251, 191, 36, 1)',
            glowClass: 'bg-yellow-400/10',
            badgeClass: 'bg-yellow-400/20 text-yellow-400',
            borderClass: 'border-yellow-400/30',
            description: 'Analysis results are inconclusive. Manual expert review is recommended.'
        },
        'LIKELY_REAL': {
            text: 'LIKELY AUTHENTIC',
            circleColor: 'rgba(34, 197, 94, 1)',
            glowClass: 'bg-accent-green/10',
            badgeClass: 'bg-accent-green/20 text-accent-green',
            borderClass: 'border-accent-green/30',
            description: 'This image shows minor inconsistencies but maintains reasonable integrity in metadata and pixel structure.'
        },
        'REAL': {
            text: 'AUTHENTIC',
            circleColor: 'rgba(0, 217, 145, 1)',
            glowClass: 'bg-accent-green/10',
            badgeClass: 'bg-accent-green/20 text-accent-green',
            borderClass: 'border-accent-green/30',
            description: 'This image appears authentic with consistent metadata, natural noise patterns, and no AI generation signatures detected.'
        }
    };

    return configs[verdict] || configs['UNCERTAIN'];
}

/**
 * Escape HTML to prevent XSS
 */
function escapeHTML(str) {
    if (!str) return '';
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

/**
 * Update status badge
 */
function updateStatusBadge(text, type = 'analyzing') {
    const statusBadge = document.getElementById('statusBadge');
    if (!statusBadge) return;

    const classes = {
        analyzing: 'px-2 py-0.5 rounded text-[10px] font-bold bg-primary/10 text-primary border border-primary/20 animate-pulse',
        success: 'px-2 py-0.5 rounded text-[10px] font-bold bg-accent-success/10 text-accent-success border border-accent-success/20',
        error: 'px-2 py-0.5 rounded text-[10px] font-bold bg-accent-danger/10 text-accent-danger border border-accent-danger/20',
        complete: 'px-2 py-0.5 rounded text-[10px] font-bold bg-accent-green/10 text-accent-green border border-accent-green/20'
    };

    statusBadge.textContent = text;
    statusBadge.className = classes[type] || classes.analyzing;
}

/**
 * Show loading state
 */
function showLoadingState() {
    updateStatusBadge('ANALYZING...', 'analyzing');
}

/**
 * Hide loading state
 */
function hideLoadingState() {
    updateStatusBadge('COMPLETED', 'complete');
}

/**
 * Display error message
 */
function displayError(message) {
    const tabContent = document.getElementById('tab-content');
    if (tabContent) {
        tabContent.innerHTML = `
            <div class="flex items-center justify-center min-h-[400px]">
                <div class="text-center max-w-md">
                    <span class="material-symbols-outlined text-red-500 text-6xl mb-4 block">error</span>
                    <h3 class="text-white text-xl font-semibold mb-2">Analysis Error</h3>
                    <p class="text-slate-400 text-sm">${escapeHTML(message)}</p>
                </div>
            </div>
        `;
    }
}

/**
 * Display failed state
 */
function displayFailedState() {
    const statusBadge = document.getElementById('statusBadge');
    if (statusBadge) {
        statusBadge.textContent = 'FAILED';
        statusBadge.className = 'px-2 py-0.5 rounded text-[10px] font-bold bg-red-500/10 text-red-400 border border-red-500/20';
    }
}

// ============================================================================
// Global Function Exposure for HTML onclick handlers
// ============================================================================

window.switchTab = switchTab;
window.switchVisualization = switchVisualization;
window.openFullscreenViewer = openFullscreenViewer;
window.closeFullscreenViewer = closeFullscreenViewer;
window.openFeedbackModal = openFeedbackModal;
window.closeFeedbackModal = closeFeedbackModal;
window.submitFeedback = submitFeedback;
window.exportToPDF = exportToPDF;
window.handleReanalyze = handleReanalyze;
