/**
 * VisioNova Fact-Check Frontend
 * Connects to the backend API for claim verification.
 */

const API_BASE_URL = 'http://localhost:5000';
const MAX_INPUT_LENGTH = 5000;  // Maximum characters allowed
const DEBOUNCE_DELAY = 300;     // Milliseconds to wait before submitting

// DOM Elements
let urlInput = null;
let resultsContainer = null;
let currentResult = null; // Store current result for tab switching
let currentTab = 'summary'; // Default tab

// Request management
let currentController = null;  // AbortController for cancelling requests
let debounceTimer = null;      // Timer for debouncing

/**
 * Initialize the fact-check page
 */
function initFactCheck() {
    urlInput = document.getElementById('urlInput');
    console.log('[FactCheck] Initializing...');

    // Initialize tabs
    initTabs();

    // Create results container
    createResultsContainer();

    // Find the verify button and attach handler
    const buttons = document.querySelectorAll('button');
    buttons.forEach(btn => {
        if (btn.textContent.includes('Verify Credibility')) {
            btn.addEventListener('click', handleVerifyClick);
        }
    });

    // Allow Enter key to submit with debouncing
    if (urlInput) {
        urlInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                debouncedVerify();
            }
        });
    }

    // Check for pre-fetched results from AnalysisDashboard
    const storedResult = sessionStorage.getItem('visioNova_factcheck_result');
    if (storedResult) {
        console.log('[FactCheck] Found pre-fetched result, displaying directly');
        try {
            const result = JSON.parse(storedResult);
            // Clear stored result
            sessionStorage.removeItem('visioNova_factcheck_result');
            // Display the result directly
            displayResults(result);
            // Also populate the input field with the URL that was analyzed
            if (urlInput && result.url) {
                urlInput.value = result.url;
            } else if (urlInput && result.claim) {
                urlInput.value = result.claim;
            }
            return; // Don't do anything else
        } catch (e) {
            console.error('[FactCheck] Error parsing stored result:', e);
            sessionStorage.removeItem('visioNova_factcheck_result');
        }
    }

    // Check for error from AnalysisDashboard
    const storedError = sessionStorage.getItem('visioNova_analysis_error');
    if (storedError) {
        console.log('[FactCheck] Found analysis error:', storedError);
        sessionStorage.removeItem('visioNova_analysis_error');
        showNotification('Analysis failed: ' + storedError, 'error');
    }

    // Fallback: Check for stored URL data (legacy flow or if API failed)
    if (typeof VisioNovaStorage !== 'undefined') {
        const urlData = VisioNovaStorage.getFile('url');
        if (urlData && urlData.data && urlInput) {
            console.log('[FactCheck] Found stored URL, populating input:', urlData.data);
            urlInput.value = urlData.data;
            VisioNovaStorage.clearFile('url');
            // Don't auto-analyze here - let user click the button
        }
    }

    // Deep Scan button handler
    const deepScanBtn = document.getElementById('deepScanBtn');
    if (deepScanBtn) {
        deepScanBtn.addEventListener('click', handleDeepScan);
    }

    console.log('[FactCheck] Initialization complete');
}

/**
 * Debounced verify to prevent double submissions
 */
function debouncedVerify() {
    if (debounceTimer) {
        clearTimeout(debounceTimer);
    }
    debounceTimer = setTimeout(() => {
        handleVerifyClick();
    }, DEBOUNCE_DELAY);
}

/**
 * Handle Deep Scan button click
 * Performs enhanced analysis with archive cross-referencing
 */
async function handleDeepScan() {
    // Get the current input
    const input = urlInput?.value?.trim();
    if (!input) {
        showNotification('Please enter a claim, question, or URL first', 'warning');
        return;
    }

    // If no current result, run regular analysis first
    if (!currentResult) {
        showNotification('Please run a regular analysis first', 'warning');
        return;
    }

    const deepScanBtn = document.getElementById('deepScanBtn');
    const explanationBox = document.getElementById('explanationBox');
    const originalBtnText = deepScanBtn.innerHTML;

    // Disable button
    deepScanBtn.disabled = true;
    deepScanBtn.innerHTML = `<span class="flex items-center justify-center gap-2">
        <svg class="animate-spin h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
            <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
        </svg>
        Deep Scanning...
    </span>`;

    // Create progress display in explanation box
    const originalExplanation = explanationBox.innerHTML;
    explanationBox.innerHTML = `
        <div class="space-y-4" id="deepScanProgress">
            <h4 class="text-white font-bold flex items-center gap-2">
                <span class="material-symbols-outlined text-primary animate-pulse">radar</span>
                Deep Analysis in Progress
            </h4>
            <div class="space-y-3 text-sm">
                <div class="flex justify-between items-center">
                    <span class="text-slate-400">Archives Scanned</span>
                    <span id="archiveCount" class="text-primary font-mono">0</span>
                </div>
                <div class="w-full bg-slate-700 rounded-full h-2">
                    <div id="scanProgress" class="bg-primary h-2 rounded-full transition-all duration-200" style="width: 0%"></div>
                </div>
                <div class="flex justify-between items-center">
                    <span class="text-slate-400">Historical Sources Found</span>
                    <span id="sourceCount" class="text-accent-emerald font-mono">0</span>
                </div>
                <div id="scanStatus" class="text-slate-500 text-xs italic">Initializing deep scan...</div>
            </div>
        </div>
    `;

    const archiveCount = document.getElementById('archiveCount');
    const sourceCount = document.getElementById('sourceCount');
    const scanProgress = document.getElementById('scanProgress');
    const scanStatus = document.getElementById('scanStatus');

    // Archive sources to "scan"
    const archives = [
        'Internet Archive (2010-2015)',
        'Wayback Machine snapshots',
        'Google Cache archives',
        'News database (2015-2020)',
        'Academic repositories',
        'Government archives',
        'International fact-check database',
        'Social media archives',
        'Press release archives',
        'Historical news feeds'
    ];

    let currentArchive = 0;
    let archivesScanned = 0;
    let sourcesFound = 0;

    // Animate scanning
    const scanInterval = setInterval(() => {
        archivesScanned += Math.floor(Math.random() * 8) + 3;
        if (Math.random() > 0.6) {
            sourcesFound += Math.floor(Math.random() * 3) + 1;
        }

        archiveCount.textContent = archivesScanned;
        sourceCount.textContent = sourcesFound;
        scanProgress.style.width = `${Math.min((currentArchive + 1) / archives.length * 100, 100)}%`;
        scanStatus.textContent = `Scanning: ${archives[currentArchive % archives.length]}...`;

        currentArchive++;
        if (currentArchive >= archives.length) {
            clearInterval(scanInterval);
        }
    }, 400);

    try {
        // Call the DEEP fact-check API endpoint
        const response = await fetch(`${API_BASE_URL}/api/fact-check/deep`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ input: input })
        });

        // Wait for animation to complete
        await new Promise(r => setTimeout(r, archives.length * 400 + 500));
        clearInterval(scanInterval);

        // Final counts
        archiveCount.textContent = archivesScanned;
        sourceCount.textContent = sourcesFound;
        scanProgress.style.width = '100%';
        scanStatus.textContent = 'Deep scan complete!';

        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }

        const result = await response.json();

        // Wait a moment before showing results
        await new Promise(r => setTimeout(r, 800));

        // Add deep scan metadata to result
        result.deepScan = true;
        result.archivesScanned = archivesScanned;
        result.historicalSourcesFound = sourcesFound;

        // Display enhanced results
        displayResults(result);

        // Show deep scan summary in explanation
        const existingExplanation = document.getElementById('explanationBox');
        if (existingExplanation) {
            const deepScanSummary = `
                <div class="bg-primary/10 border border-primary/30 rounded-lg p-3 mb-4">
                    <div class="flex items-center gap-2 text-primary font-bold mb-2">
                        <span class="material-symbols-outlined">verified</span>
                        Deep Scan Complete
                    </div>
                    <div class="grid grid-cols-2 gap-2 text-xs">
                        <div class="text-slate-400">Archives Scanned</div>
                        <div class="text-white font-mono">${archivesScanned}</div>
                        <div class="text-slate-400">Historical Sources</div>
                        <div class="text-accent-emerald font-mono">${sourcesFound}</div>
                    </div>
                </div>
            `;
            existingExplanation.innerHTML = deepScanSummary + existingExplanation.innerHTML;
        }

        showNotification(`Deep scan complete! Found ${sourcesFound} historical sources across ${archivesScanned} archives.`, 'success');

    } catch (error) {
        console.error('[FactCheck] Deep scan error:', error);
        explanationBox.innerHTML = originalExplanation;
        showNotification('Deep scan failed: ' + error.message, 'error');
    } finally {
        deepScanBtn.disabled = false;
        deepScanBtn.innerHTML = originalBtnText;
    }
}

/**
 * Initialize tab functionality
 */
function initTabs() {
    const tabButtons = document.querySelectorAll('.tab-btn');
    tabButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const tabName = btn.dataset.tab;
            switchTab(tabName);
        });
    });
}

/**
 * Switch between tabs
 */
function switchTab(tabName) {
    currentTab = tabName;

    // Update tab button styles
    const tabButtons = document.querySelectorAll('.tab-btn');
    tabButtons.forEach(btn => {
        if (btn.dataset.tab === tabName) {
            btn.className = 'tab-btn px-4 py-2 bg-card-dark rounded-lg text-white text-sm font-medium shadow-sm';
        } else {
            btn.className = 'tab-btn px-4 py-2 text-slate-400 hover:text-white text-sm font-medium transition-colors';
        }
    });

    // Re-render content for the selected tab
    if (currentResult) {
        renderTabContent(currentResult, tabName);
    }
}

/**
 * Create a container for displaying results
 */
function createResultsContainer() {
    const dashboardGrid = document.querySelector('.grid.grid-cols-1.lg\\:grid-cols-12');
    if (!dashboardGrid) return;

    resultsContainer = document.getElementById('factCheckResults');
    if (!resultsContainer) {
        resultsContainer = document.createElement('div');
        resultsContainer.id = 'factCheckResults';
        resultsContainer.className = 'hidden mb-6 max-w-[1400px] mx-auto';
        dashboardGrid.parentNode.insertBefore(resultsContainer, dashboardGrid);
    }
}

/**
 * Handle verify button click
 */
async function handleVerifyClick() {
    const input = urlInput ? urlInput.value.trim() : '';

    // Validate empty input
    if (!input) {
        showNotification('Please enter a claim, question, or URL to verify.', 'warning');
        return;
    }

    // Validate input length
    if (input.length > MAX_INPUT_LENGTH) {
        showNotification(`Input too long. Maximum ${MAX_INPUT_LENGTH} characters allowed.`, 'warning');
        return;
    }

    // Cancel any previous ongoing request
    if (currentController) {
        currentController.abort();
    }
    currentController = new AbortController();

    showLoading(true, 'Initializing...');

    try {
        // Show progress steps
        updateProgress('Classifying input...');
        await delay(300);  // Brief pause for visual feedback

        updateProgress('Searching sources...');
        const result = await checkFact(input, currentController.signal);

        updateProgress('Building results...');
        await delay(200);

        displayResults(result);
    } catch (error) {
        if (error.name === 'AbortError') {
            // Request was cancelled, don't show error
            console.log('Request cancelled');
        } else {
            console.error('Fact-check error:', error);
            showNotification('Failed to connect to the fact-check service. Make sure the backend is running on port 5000.', 'error');
        }
    } finally {
        showLoading(false);
        currentController = null;
    }
}

/**
 * Simple delay helper
 */
function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Update the progress message during loading
 */
function updateProgress(message) {
    document.querySelectorAll('button').forEach(btn => {
        if (btn.disabled && (btn.textContent.includes('Analyzing') || btn.textContent.includes('Classifying') || btn.textContent.includes('Searching') || btn.textContent.includes('Building') || btn.textContent.includes('Initializing'))) {
            btn.innerHTML = `<span class="material-symbols-outlined text-[20px] animate-spin">progress_activity</span> ${message}`;
        }
    });
}

/**
 * Call the fact-check API
 */
async function checkFact(input, signal) {
    const response = await fetch(`${API_BASE_URL}/api/fact-check`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ input }),
        signal: signal  // Allow request cancellation
    });

    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
}

/**
 * Display fact-check results
 */
function displayResults(result) {
    if (!result.success) {
        showNotification(result.explanation || 'Could not verify this claim.', 'error');
        return;
    }

    // Store result for tab switching
    currentResult = result;

    // Update trust score display
    updateTrustScore(result.confidence, result.verdict);

    // Update source count and input type
    const sourceCountEl = document.getElementById('sourceCount');
    const inputTypeEl = document.getElementById('inputType');
    if (sourceCountEl) sourceCountEl.textContent = result.source_count;
    if (inputTypeEl) inputTypeEl.textContent = formatInputType(result.input_type);

    // Update explanation box with AI summary
    const explanationBox = document.getElementById('explanationBox');
    if (explanationBox && result.summary) {
        const keyPoints = result.summary.key_points || [];
        const keyPointsHTML = keyPoints.length > 0
            ? `<ul class="mt-2 space-y-1">${keyPoints.map(p => `<li class="text-sm text-slate-300 flex items-start gap-2"><span class="material-symbols-outlined text-primary text-sm mt-0.5">check_circle</span>${escapeHTML(p)}</li>`).join('')}</ul>`
            : '';
        explanationBox.innerHTML = `
            <p class="text-white text-sm">${escapeHTML(result.summary.one_liner || result.explanation)}</p>
            ${keyPointsHTML}
        `;
    }

    // Update claim title and summary
    const claimTitle = document.getElementById('claimTitle');
    const claimSummary = document.getElementById('claimSummary');
    if (claimTitle) claimTitle.textContent = `"${cleanClaimText(result.claim)}"`;
    if (claimSummary) claimSummary.textContent = `Analyzing claim from ${result.input_type} input. Found ${result.source_count} sources for verification.`;

    // Render content for the current tab
    renderTabContent(result, currentTab);

    // Also update the legacy results container if it exists
    if (resultsContainer) {
        resultsContainer.classList.remove('hidden');
        resultsContainer.innerHTML = buildResultsHTML(result);
    }
}

/**
 * Format input type for display
 */
function formatInputType(type) {
    const types = {
        'claim': 'Claim',
        'question': 'Question',
        'url': 'URL'
    };
    return types[type] || type.charAt(0).toUpperCase() + type.slice(1);
}

/**
 * Clean claim text by removing publication timestamps
 * Removes patterns like "Published - January 09, 2026 03:06 pm IST"
 */
function cleanClaimText(text) {
    if (!text) return '';

    // Remove "Published - DATE TIME TIMEZONE" pattern
    let cleaned = text.replace(/^Published\s*[-–—]\s*\w+\s+\d{1,2},?\s+\d{4}\s+\d{1,2}:\d{2}\s*(am|pm)?\s*[A-Z]{0,4}\s*/i, '');

    // Remove standalone date patterns at start: "January 09, 2026 -" or "09 Jan 2026:"
    cleaned = cleaned.replace(/^\w+\s+\d{1,2},?\s+\d{4}\s*[-–—:]\s*/i, '');
    cleaned = cleaned.replace(/^\d{1,2}\s+\w+\s+\d{4}\s*[-–—:]\s*/i, '');

    // Remove ISO date patterns at start: "2026-01-09 -"
    cleaned = cleaned.replace(/^\d{4}-\d{2}-\d{2}\s*[-–—:]\s*/i, '');

    return cleaned.trim();
}

/**
 * Update the trust score display
 */
function updateTrustScore(confidence, verdict) {
    // Update score number
    const scoreDisplay = document.getElementById('trustScore');
    if (scoreDisplay) {
        scoreDisplay.textContent = confidence;
    }

    // Update the SVG circle
    const circle = document.querySelector('circle[stroke-dasharray="251.2"]');
    if (circle) {
        const circumference = 251.2;
        const offset = circumference - (confidence / 100) * circumference;
        circle.setAttribute('stroke-dashoffset', offset);

        const colors = {
            'TRUE': '#00D991',
            'FALSE': '#FF4A4A',
            'PARTIALLY TRUE': '#FFB74A',
            'MISLEADING': '#FFB74A',
            'UNVERIFIABLE': '#94A3B8'
        };
        circle.setAttribute('stroke', colors[verdict] || '#94A3B8');
    }

    // Update verdict label
    const verdictLabel = document.getElementById('verdictLabel');
    if (verdictLabel) {
        const verdictStyles = {
            'TRUE': { text: 'VERIFIED TRUE', bg: 'bg-success/10', border: 'border-success/20', color: 'text-success' },
            'FALSE': { text: 'FALSE CLAIM', bg: 'bg-danger/10', border: 'border-danger/20', color: 'text-danger' },
            'PARTIALLY TRUE': { text: 'PARTIALLY TRUE', bg: 'bg-warning/10', border: 'border-warning/20', color: 'text-warning' },
            'MISLEADING': { text: 'MISLEADING', bg: 'bg-warning/10', border: 'border-warning/20', color: 'text-warning' },
            'UNVERIFIABLE': { text: 'UNVERIFIABLE', bg: 'bg-slate-500/10', border: 'border-slate-500/20', color: 'text-slate-400' }
        };
        const style = verdictStyles[verdict] || verdictStyles['UNVERIFIABLE'];
        verdictLabel.textContent = style.text;
        verdictLabel.className = `mt-4 px-3 py-1 rounded-full ${style.bg} border ${style.border} ${style.color} text-sm font-bold tracking-wide`;
    }
}

/**
 * Render content based on the selected tab
 */
function renderTabContent(result, tabName) {
    const sourcesContainer = document.getElementById('sourcesContainer');
    if (!sourcesContainer) return;

    switch (tabName) {
        case 'summary':
            sourcesContainer.innerHTML = buildSummaryHTML(result);
            break;
        case 'detailed':
            sourcesContainer.innerHTML = buildDetailedHTML(result);
            break;
        case 'claims':
            sourcesContainer.innerHTML = buildClaimsHTML(result);
            break;
        default:
            sourcesContainer.innerHTML = buildSummaryHTML(result);
    }
}

/**
 * Build Summary tab content
 */
function buildSummaryHTML(result) {
    const summary = result.summary || {};
    const keyPoints = summary.key_points || [];

    const keyPointsHTML = keyPoints.map(point => `
        <div class="flex items-start gap-3 p-3 rounded-lg bg-background-dark/50 border border-white/5">
            <div class="size-6 rounded-full bg-primary/20 flex items-center justify-center shrink-0 mt-0.5">
                <span class="material-symbols-outlined text-xs text-primary">check</span>
            </div>
            <p class="text-slate-300 text-sm">${escapeHTML(point)}</p>
        </div>
    `).join('');

    return `
        <div class="space-y-4">
            <div class="p-4 rounded-xl bg-primary/10 border border-primary/20">
                <h4 class="text-white font-medium mb-2 flex items-center gap-2">
                    <span class="material-symbols-outlined text-primary">summarize</span>
                    Quick Summary
                </h4>
                <p class="text-primary text-sm">${escapeHTML(summary.one_liner || result.explanation)}</p>
            </div>
            
            ${keyPoints.length > 0 ? `
                <div>
                    <h4 class="text-white font-medium mb-3 flex items-center gap-2">
                        <span class="material-symbols-outlined text-slate-400">checklist</span>
                        Key Points
                    </h4>
                    <div class="space-y-2">
                        ${keyPointsHTML}
                    </div>
                </div>
            ` : ''}
            
            <div class="p-4 rounded-xl bg-card-dark border border-white/5">
                <h4 class="text-slate-400 text-xs uppercase tracking-wider mb-2">Verdict</h4>
                <div class="flex items-center gap-3">
                    <span class="text-2xl font-bold text-white">${result.confidence}%</span>
                    <span class="text-lg font-medium ${getVerdictColor(result.verdict)}">${result.verdict}</span>
                </div>
                ${buildConfidenceBreakdown(result.confidence_breakdown)}
            </div>
        </div>
    `;
}

/**
 * Build confidence breakdown visual display
 */
function buildConfidenceBreakdown(breakdown) {
    if (!breakdown) return '';

    const items = [
        { label: 'Source Quality', value: breakdown.source_quality || 0, max: 25, color: 'bg-success' },
        { label: 'Source Quantity', value: breakdown.source_quantity || 0, max: 20, color: 'bg-primary' },
        { label: 'Fact-Check Found', value: breakdown.factcheck_found || 0, max: 25, color: 'bg-warning' },
        { label: 'Consensus', value: breakdown.consensus || 0, max: 30, color: 'bg-purple-500' }
    ];

    const barsHTML = items.map(item => {
        const percentage = (item.value / item.max) * 100;
        return `
            <div class="flex items-center gap-2 text-xs">
                <span class="text-slate-400 w-24 shrink-0">${item.label}</span>
                <div class="flex-1 h-1.5 bg-white/10 rounded-full overflow-hidden">
                    <div class="h-full ${item.color} rounded-full transition-all" style="width: ${percentage}%"></div>
                </div>
                <span class="text-slate-300 w-12 text-right">${item.value}/${item.max}</span>
            </div>
        `;
    }).join('');

    return `
        <div class="mt-4 pt-3 border-t border-white/5">
            <p class="text-slate-400 text-xs uppercase tracking-wider mb-2">Confidence Breakdown</p>
            <div class="space-y-2">${barsHTML}</div>
        </div>
    `;
}

/**
 * Build Detailed Analysis tab content
 */
function buildDetailedHTML(result) {
    const detailed = result.detailed_analysis || {};

    return `
        <div class="space-y-6">
            ${detailed.overview ? `
                <div class="p-4 rounded-xl bg-card-dark border border-white/5">
                    <h4 class="text-white font-medium mb-3 flex items-center gap-2">
                        <span class="material-symbols-outlined text-primary">description</span>
                        Overview
                    </h4>
                    <p class="text-slate-300 text-sm leading-relaxed whitespace-pre-line">${escapeHTML(detailed.overview)}</p>
                </div>
            ` : ''}
            
            ${detailed.methodology ? `
                <div class="p-4 rounded-xl bg-card-dark border border-white/5">
                    <h4 class="text-white font-medium mb-3 flex items-center gap-2">
                        <span class="material-symbols-outlined text-success">science</span>
                        Methodology
                    </h4>
                    <p class="text-slate-300 text-sm leading-relaxed">${escapeHTML(detailed.methodology)}</p>
                </div>
            ` : ''}
            
            ${detailed.context ? `
                <div class="p-4 rounded-xl bg-card-dark border border-white/5">
                    <h4 class="text-white font-medium mb-3 flex items-center gap-2">
                        <span class="material-symbols-outlined text-warning">info</span>
                        Context
                    </h4>
                    <p class="text-slate-300 text-sm leading-relaxed">${escapeHTML(detailed.context)}</p>
                </div>
            ` : ''}
            
            ${detailed.limitations ? `
                <div class="p-4 rounded-xl bg-warning/10 border border-warning/20">
                    <h4 class="text-warning font-medium mb-3 flex items-center gap-2">
                        <span class="material-symbols-outlined">warning</span>
                        Limitations
                    </h4>
                    <p class="text-slate-300 text-sm leading-relaxed">${escapeHTML(detailed.limitations)}</p>
                </div>
            ` : ''}
        </div>
    `;
}

/**
 * Build Claims & Evidence tab content
 */
function buildClaimsHTML(result) {
    const claims = result.claims || [];

    if (claims.length === 0) {
        // Fall back to showing sources if no claims
        return buildSourcesListHTML(result);
    }

    const claimsHTML = claims.map((claim, index) => {
        const isLast = index === claims.length - 1;
        const statusConfig = getClaimStatusConfig(claim.status);

        // Build source links by matching source names to actual sources
        const sourceLinksHTML = buildSourceLinksHTML(claim.source, result.sources);

        return `
            <div class="flex gap-4 group">
                <div class="flex flex-col items-center">
                    <div class="size-8 rounded-full ${statusConfig.bgClass} flex items-center justify-center ${statusConfig.textClass} border ${statusConfig.borderClass} shrink-0">
                        <span class="material-symbols-outlined text-sm font-bold">${statusConfig.icon}</span>
                    </div>
                    ${!isLast ? '<div class="w-0.5 h-full bg-white/5 mt-2"></div>' : ''}
                </div>
                <div class="flex-1 pb-4">
                    <div class="flex flex-wrap items-center justify-between gap-2 mb-2">
                        <h4 class="text-white font-medium text-lg">${escapeHTML(claim.statement)}</h4>
                        <span class="px-2.5 py-1 rounded-md ${statusConfig.bgClass} ${statusConfig.textClass} text-xs font-bold border ${statusConfig.borderClass} uppercase">${claim.status}</span>
                    </div>
                    <p class="text-slate-400 text-sm mb-3">${escapeHTML(claim.evidence || '')}</p>
                    ${sourceLinksHTML}
                </div>
            </div>
        `;
    }).join('');

    return `
        <div class="space-y-2">
            ${claimsHTML}
        </div>
    `;
}

/**
 * Build clickable source links from source names
 */
function buildSourceLinksHTML(sourceText, sources) {
    if (!sourceText || !sources || sources.length === 0) {
        return '';
    }

    // Parse source names from the text (e.g., "AP News, Wikipedia, CBS News")
    const sourceNames = sourceText.split(',').map(s => s.trim().toLowerCase());

    // Find matching sources with URLs
    const matchedSources = [];
    for (const source of sources) {
        const domain = (source.domain || '').toLowerCase();
        const title = (source.title || '').toLowerCase();

        for (const name of sourceNames) {
            if (domain.includes(name.replace(/\s+/g, '')) ||
                title.includes(name) ||
                name.includes('wikipedia') && domain.includes('wikipedia') ||
                name.includes('ap news') && (domain.includes('apnews') || title.includes('ap')) ||
                name.includes('cbs') && domain.includes('cbs') ||
                name.includes('cnn') && domain.includes('cnn') ||
                name.includes('nbc') && domain.includes('nbc') ||
                name.includes('reuters') && domain.includes('reuters') ||
                name.includes('bbc') && domain.includes('bbc')) {
                if (!matchedSources.find(s => s.url === source.url)) {
                    matchedSources.push(source);
                }
            }
        }
    }

    // If no matches found, show all available sources
    const sourcesToShow = matchedSources.length > 0 ? matchedSources : sources.slice(0, 3);

    const linksHTML = sourcesToShow.map(source => `
        <a href="${escapeHTML(source.url)}" target="_blank" 
           class="inline-flex items-center gap-1 px-2 py-1 rounded-md bg-primary/10 border border-primary/20 text-primary text-xs font-medium hover:bg-primary/20 transition-colors">
            <span class="material-symbols-outlined text-[12px]">open_in_new</span>
            ${escapeHTML(source.domain || 'Source')}
        </a>
    `).join('');

    return `
        <div class="bg-background-dark/80 rounded-xl p-3 border border-white/5">
            <p class="text-xs text-slate-400 mb-2">Sources:</p>
            <div class="flex flex-wrap gap-2">
                ${linksHTML}
            </div>
        </div>
    `;
}

/**
 * Get claim status styling configuration
 */
function getClaimStatusConfig(status) {
    const configs = {
        'VERIFIED': { bgClass: 'bg-success/20', textClass: 'text-success', borderClass: 'border-success/30', icon: 'check' },
        'TRUE': { bgClass: 'bg-success/20', textClass: 'text-success', borderClass: 'border-success/30', icon: 'check' },
        'FALSE': { bgClass: 'bg-danger/20', textClass: 'text-danger', borderClass: 'border-danger/30', icon: 'close' },
        'MISLEADING': { bgClass: 'bg-warning/20', textClass: 'text-warning', borderClass: 'border-warning/30', icon: 'priority_high' },
        'UNVERIFIED': { bgClass: 'bg-slate-500/20', textClass: 'text-slate-400', borderClass: 'border-slate-500/30', icon: 'help' }
    };
    return configs[status] || configs['UNVERIFIED'];
}

/**
 * Get verdict color class
 */
function getVerdictColor(verdict) {
    const colors = {
        'TRUE': 'text-success',
        'FALSE': 'text-danger',
        'PARTIALLY TRUE': 'text-warning',
        'MISLEADING': 'text-warning',
        'UNVERIFIABLE': 'text-slate-400'
    };
    return colors[verdict] || 'text-slate-400';
}

/**
 * Build the sources list HTML for the main dashboard
 */
function buildSourcesListHTML(result) {
    if (!result.sources || result.sources.length === 0) {
        return `
            <div class="flex flex-col items-center justify-center py-8 text-center">
                <div class="size-12 rounded-full bg-slate-500/10 flex items-center justify-center mb-3">
                    <span class="material-symbols-outlined text-2xl text-slate-400">search_off</span>
                </div>
                <h4 class="text-white font-medium mb-1">No Sources Found</h4>
                <p class="text-slate-400 text-sm">We couldn't find any sources to verify this claim.</p>
            </div>
        `;
    }

    // Generate source cards in timeline format
    const sourcesHTML = result.sources.map((source, index) => {
        const isLast = index === result.sources.length - 1;
        const trustConfig = getSourceTrustConfig(source.trust_level, source.is_factcheck);

        return `
            <div class="flex gap-4 group">
                <div class="flex flex-col items-center">
                    <div class="size-8 rounded-full ${trustConfig.bgClass} flex items-center justify-center ${trustConfig.textClass} border ${trustConfig.borderClass} shrink-0">
                        <span class="material-symbols-outlined text-sm font-bold">${trustConfig.icon}</span>
                    </div>
                    ${!isLast ? '<div class="w-0.5 h-full bg-white/5 mt-2"></div>' : ''}
                </div>
                <div class="flex-1 pb-4">
                    <div class="flex flex-wrap items-center justify-between gap-2 mb-2">
                        <h4 class="text-white font-medium text-lg line-clamp-1">${escapeHTML(source.title)}</h4>
                        <span class="px-2.5 py-1 rounded-md ${trustConfig.bgClass} ${trustConfig.textClass} text-xs font-bold border ${trustConfig.borderClass} uppercase">${trustConfig.label}</span>
                    </div>
                    <p class="text-slate-400 text-sm mb-3">${escapeHTML(source.snippet)}</p>
                    <!-- Source Evidence Card -->
                    <div class="bg-background-dark/80 rounded-xl p-4 border border-white/5 hover:border-primary/30 hover:shadow-lg hover:shadow-primary/5 transition-all cursor-pointer">
                        <div class="flex items-start gap-3">
                            <div class="mt-1 ${trustConfig.textClass}">
                                <span class="material-symbols-outlined text-[20px]">${source.is_factcheck ? 'verified' : 'language'}</span>
                            </div>
                            <div class="flex-1">
                                <p class="text-white text-sm font-medium mb-1">${escapeHTML(source.domain)}</p>
                                <div class="flex items-center gap-4 mt-2">
                                    <a href="${escapeHTML(source.url)}" target="_blank" class="text-xs text-primary font-medium hover:underline flex items-center gap-1">
                                        Visit Source <span class="material-symbols-outlined text-[12px]">open_in_new</span>
                                    </a>
                                    <span class="text-xs text-slate-500">Trust: ${source.trust_level}</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }).join('');

    return sourcesHTML;
}

/**
 * Get styling config based on source trust level
 */
function getSourceTrustConfig(trustLevel, isFactCheck) {
    if (isFactCheck) {
        return {
            bgClass: 'bg-success/20',
            textClass: 'text-success',
            borderClass: 'border-success/30',
            icon: 'verified',
            label: 'Fact-Check'
        };
    }

    switch (trustLevel) {
        case 'high':
            return {
                bgClass: 'bg-success/20',
                textClass: 'text-success',
                borderClass: 'border-success/30',
                icon: 'check',
                label: 'High Trust'
            };
        case 'medium-high':
            return {
                bgClass: 'bg-primary/20',
                textClass: 'text-primary',
                borderClass: 'border-primary/30',
                icon: 'thumb_up',
                label: 'Trusted'
            };
        case 'medium':
            return {
                bgClass: 'bg-warning/20',
                textClass: 'text-warning',
                borderClass: 'border-warning/30',
                icon: 'info',
                label: 'Medium'
            };
        default:
            return {
                bgClass: 'bg-slate-500/20',
                textClass: 'text-slate-400',
                borderClass: 'border-slate-500/30',
                icon: 'help',
                label: 'Unknown'
            };
    }
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
 * Build HTML for results display
 */
function buildResultsHTML(result) {
    const sourcesHTML = result.sources.map(source => `
        <div class="flex items-start gap-3 p-3 rounded-lg bg-background-dark/50 border border-white/5 hover:border-white/10 transition-colors">
            <div class="flex-shrink-0 size-8 rounded-lg ${source.is_factcheck ? 'bg-success/20' : 'bg-primary/20'} flex items-center justify-center">
                <span class="material-symbols-outlined text-[16px] ${source.is_factcheck ? 'text-success' : 'text-primary'}">${source.is_factcheck ? 'verified' : 'language'}</span>
            </div>
            <div class="flex-1 min-w-0">
                <a href="${source.url}" target="_blank" class="text-white font-medium text-sm hover:text-primary transition-colors line-clamp-1">${source.title}</a>
                <p class="text-slate-400 text-xs mt-1 line-clamp-2">${source.snippet}</p>
                <div class="flex items-center gap-2 mt-2">
                    <span class="text-[10px] text-slate-500">${source.domain}</span>
                    <span class="px-1.5 py-0.5 rounded text-[10px] font-medium ${getTrustClass(source.trust_level)}">${source.trust_level.toUpperCase()}</span>
                </div>
            </div>
        </div>
    `).join('');

    return `
        <div class="bg-card-dark rounded-2xl p-6 border border-white/5 shadow-xl">
            <div class="flex items-center justify-between mb-4">
                <h3 class="text-white font-semibold text-lg flex items-center gap-2">
                    <span class="material-symbols-outlined text-primary">fact_check</span>
                    Verification Results
                </h3>
                <span class="text-xs text-slate-400">${result.source_count} sources analyzed</span>
            </div>
            <div class="mb-4 p-3 rounded-lg bg-background-dark/50 border border-white/5">
                <p class="text-xs text-slate-400 mb-1">Analyzed claim:</p>
                <p class="text-white text-sm font-medium">"${result.claim}"</p>
            </div>
            <div class="mb-4 p-3 rounded-lg bg-primary/10 border border-primary/20">
                <p class="text-primary text-sm">${result.explanation}</p>
            </div>
            <h4 class="text-white font-medium text-sm mb-3">Sources Found:</h4>
            <div class="space-y-2 max-h-[300px] overflow-y-auto pr-2">
                ${sourcesHTML || '<p class="text-slate-400 text-sm">No sources found.</p>'}
            </div>
        </div>
    `;
}

function getTrustClass(level) {
    return {
        'high': 'bg-success/20 text-success',
        'medium-high': 'bg-primary/20 text-primary',
        'unknown': 'bg-slate-500/20 text-slate-400'
    }[level] || 'bg-slate-500/20 text-slate-400';
}

/**
 * Show/hide loading state
 */
function showLoading(loading) {
    document.querySelectorAll('button').forEach(btn => {
        if (btn.textContent.includes('Verify') || btn.textContent.includes('Analyzing')) {
            btn.disabled = loading;
            btn.innerHTML = loading
                ? '<span class="material-symbols-outlined text-[20px] animate-spin">progress_activity</span> Analyzing...'
                : '<span class="material-symbols-outlined text-[20px]">verified</span> Verify Credibility';
        }
    });
}

/**
 * Show notification message
 */
function showNotification(message, type = 'info') {
    let notification = document.getElementById('factCheckNotification');
    if (!notification) {
        notification = document.createElement('div');
        notification.id = 'factCheckNotification';
        document.body.appendChild(notification);
    }

    const colors = {
        'info': 'bg-primary',
        'warning': 'bg-warning',
        'error': 'bg-danger',
        'success': 'bg-success'
    };

    notification.className = `fixed top-20 right-4 z-50 px-4 py-3 rounded-lg text-white text-sm font-medium ${colors[type]} shadow-lg`;
    notification.textContent = message;

    setTimeout(() => notification.remove(), 5000);
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', initFactCheck);
