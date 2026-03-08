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









/**
 * Initialize the fact-check page
 */
async function initFactCheck() {
    urlInput = document.getElementById('urlInput');
    console.log('[FactCheck] Initializing...');



    // Initialize tabs
    initTabs();

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
        UI.showNotification('Analysis failed: ' + storedError, 'error');
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
        console.log('[FactCheck] Deep Scan button initialized');
    } else {
        console.warn('[FactCheck] Deep Scan button NOT FOUND - check if id="deepScanBtn" exists in HTML');
    }

    // Initialize feedback modal handlers
    initFeedbackModal();

    // Initialize PDF export button
    const exportPdfBtn = document.getElementById('exportPdfBtn');
    if (exportPdfBtn) {
        exportPdfBtn.addEventListener('click', exportToPDF);
    }

    console.log('[FactCheck] Initialization complete');
}

/**
 * Initialize feedback modal and button handlers
 */
function initFeedbackModal() {
    const feedbackAgree = document.getElementById('feedbackAgree');
    const feedbackDisagree = document.getElementById('feedbackDisagree');
    const closeFeedbackModal = document.getElementById('closeFeedbackModal');
    const cancelFeedback = document.getElementById('cancelFeedback');
    const submitFeedback = document.getElementById('submitFeedback');
    const feedbackModal = document.getElementById('feedbackModal');

    if (feedbackAgree) {
        feedbackAgree.addEventListener('click', () => {
            UI.showNotification('Thank you for your feedback!', 'success');
            // Optional: Send agreement feedback to backend
            sendFeedbackToBackend('agree');
        });
    }

    if (feedbackDisagree) {
        feedbackDisagree.addEventListener('click', () => {
            openFeedbackModal();
        });
    }

    if (closeFeedbackModal) {
        closeFeedbackModal.addEventListener('click', () => {
            feedbackModal.classList.add('hidden');
        });
    }

    if (cancelFeedback) {
        cancelFeedback.addEventListener('click', () => {
            feedbackModal.classList.add('hidden');
        });
    }

    if (submitFeedback) {
        submitFeedback.addEventListener('click', handleFeedbackSubmission);
    }

    // Close modal on outside click
    if (feedbackModal) {
        feedbackModal.addEventListener('click', (e) => {
            if (e.target === feedbackModal) {
                feedbackModal.classList.add('hidden');
            }
        });
    }
}

/**
 * Open feedback modal
 */
function openFeedbackModal() {
    const feedbackModal = document.getElementById('feedbackModal');
    const feedbackSystemVerdict = document.getElementById('feedbackSystemVerdict');

    if (!currentResult) {
        UI.showNotification('No analysis result to provide feedback on', 'warning');
        return;
    }

    if (feedbackSystemVerdict && currentResult.verdict) {
        feedbackSystemVerdict.textContent = currentResult.verdict;
    }

    if (feedbackModal) {
        feedbackModal.classList.remove('hidden');
    }
}

/**
 * Handle feedback form submission
 */
async function handleFeedbackSubmission() {
    const feedbackUserVerdict = document.getElementById('feedbackUserVerdict');
    const feedbackReason = document.getElementById('feedbackReason');
    const feedbackSources = document.getElementById('feedbackSources');
    const submitBtn = document.getElementById('submitFeedback');

    if (!currentResult) {
        UI.showNotification('No analysis result to provide feedback on', 'error');
        return;
    }

    const userVerdict = feedbackUserVerdict.value;
    const reason = feedbackReason.value.trim();
    const sourcesInput = feedbackSources.value.trim();
    const additionalSources = sourcesInput ? sourcesInput.split(',').map(s => s.trim()).filter(s => s) : [];

    // Disable submit button
    const originalBtnText = submitBtn.innerHTML;
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<span class="material-symbols-outlined text-[18px] animate-spin">progress_activity</span> Submitting...';

    try {
        const response = await fetch(`${API_BASE_URL}/api/fact-check/feedback`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                claim: currentResult.claim || currentResult.url || urlInput.value,
                original_verdict: currentResult.verdict,
                user_verdict: userVerdict,
                reason: reason,
                additional_sources: additionalSources
            })
        });

        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }

        const result = await response.json();

        UI.showNotification('Feedback submitted successfully! Thank you for helping us improve.', 'success');

        // Close modal and reset form
        document.getElementById('feedbackModal').classList.add('hidden');
        feedbackReason.value = '';
        feedbackSources.value = '';

    } catch (error) {
        console.error('[Feedback] Submission error:', error);
        UI.showNotification('Failed to submit feedback. Please try again.', 'error');
    } finally {
        submitBtn.disabled = false;
        submitBtn.innerHTML = originalBtnText;
    }
}

/**
 * Send simple feedback (agree) to backend
 */
async function sendFeedbackToBackend(type) {
    if (!currentResult) return;

    try {
        await fetch(`${API_BASE_URL}/api/fact-check/feedback`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                claim: currentResult.claim || currentResult.url || urlInput.value,
                original_verdict: currentResult.verdict,
                user_verdict: currentResult.verdict, // Same as system for agreement
                reason: 'User agreed with the verdict',
                additional_sources: []
            })
        });
    } catch (error) {
        console.error('[Feedback] Error sending agreement:', error);
    }
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
    console.log('[FactCheck] Deep Scan button clicked');

    // Get the current input
    const input = urlInput?.value?.trim();
    if (!input) {
        console.log('[FactCheck] Deep Scan: No input provided');
        UI.showNotification('Please enter a claim, question, or URL first', 'warning');
        return;
    }

    // If no current result, run regular analysis first
    if (!currentResult) {
        console.log('[FactCheck] Deep Scan: No previous result exists - need to run regular analysis first');
        UI.showNotification('Please run a regular analysis first', 'warning');
        return;
    }

    console.log('[FactCheck] Deep Scan: Starting deep analysis...');

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

    // Get temporal context for display
    const temporalDesc = currentResult.temporal_context?.description || 'multiple archives';

    explanationBox.innerHTML = `
        <div class="space-y-4" id="deepScanProgress">
            <h4 class="text-white font-bold flex items-center gap-2">
                <span class="material-symbols-outlined text-primary animate-pulse">radar</span>
                Deep Analysis in Progress
            </h4>
            <div class="space-y-3 text-sm">
                <div class="flex justify-between items-center">
                    <span class="text-slate-400">Search Queries Executed</span>
                    <span id="archiveCount" class="text-primary font-mono">...</span>
                </div>
                <div class="w-full bg-slate-700 rounded-full h-2">
                    <div id="scanProgress" class="bg-primary h-2 rounded-full transition-all duration-500 animate-pulse" style="width: 30%"></div>
                </div>
                <div class="flex justify-between items-center">
                    <span class="text-slate-400">Sources Found</span>
                    <span id="sourceCount" class="text-accent-emerald font-mono">...</span>
                </div>
                <div id="scanStatus" class="text-slate-500 text-xs italic">Searching ${temporalDesc}...</div>
            </div>
        </div>
    `;

    const archiveCount = document.getElementById('archiveCount');
    const sourceCount = document.getElementById('sourceCount');
    const scanProgress = document.getElementById('scanProgress');
    const scanStatus = document.getElementById('scanStatus');

    // Start progress animation
    let progressPercent = 30;
    const progressInterval = setInterval(() => {
        progressPercent = Math.min(progressPercent + 10, 90);
        scanProgress.style.width = `${progressPercent}%`;
    }, 800);

    try {
        // Call the DEEP fact-check API endpoint
        scanStatus.textContent = 'Executing deep search queries...';

        const response = await fetch(`${API_BASE_URL}/api/fact-check/deep`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ input: input })
        });

        clearInterval(progressInterval);

        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }

        const result = await response.json();

        // Show REAL values from API response
        const realQueriesUsed = result.queries_used || 0;
        const realTotalSources = result.total_sources_found || 0;
        const realUniqueSources = result.unique_sources || result.source_count || 0;

        // Update progress with REAL data
        scanProgress.style.width = '100%';
        archiveCount.textContent = realQueriesUsed;
        sourceCount.textContent = realUniqueSources;
        scanStatus.textContent = `Deep scan complete! Analyzed ${realTotalSources} total results, ${realUniqueSources} unique sources.`;

        // Brief pause to show final stats
        await new Promise(r => setTimeout(r, 1000));

        // Add deep scan metadata to result
        result.deepScan = true;

        // Display enhanced results
        displayResults(result);

        // Show deep scan summary in explanation with REAL values
        const existingExplanation = document.getElementById('explanationBox');
        if (existingExplanation) {
            const temporalInfo = result.temporal_context ? `
                <div class="text-slate-400">Time Period</div>
                <div class="text-white font-mono">${result.temporal_context.search_year_from || 'Current'}</div>
            ` : '';

            const deepScanSummary = `
                <div class="bg-primary/10 border border-primary/30 rounded-lg p-3 mb-4">
                    <div class="flex items-center gap-2 text-primary font-bold mb-2">
                        <span class="material-symbols-outlined">verified</span>
                        Deep Scan Complete
                    </div>
                    <div class="grid grid-cols-2 gap-2 text-xs">
                        <div class="text-slate-400">Search Queries</div>
                        <div class="text-white font-mono">${realQueriesUsed}</div>
                        <div class="text-slate-400">Total Results</div>
                        <div class="text-white font-mono">${realTotalSources}</div>
                        <div class="text-slate-400">Unique Sources</div>
                        <div class="text-accent-emerald font-mono">${realUniqueSources}</div>
                        ${temporalInfo}
                    </div>
                </div>
            `;
            existingExplanation.innerHTML = deepScanSummary + existingExplanation.innerHTML;
        }

        UI.showNotification(`Deep scan complete! ${realQueriesUsed} queries executed, ${realUniqueSources} unique sources found.`, 'success');

    } catch (error) {
        clearInterval(progressInterval);
        console.error('[FactCheck] Deep scan error:', error);
        explanationBox.innerHTML = originalExplanation;
        UI.showNotification('Deep scan failed: ' + error.message, 'error');
    } finally {
        deepScanBtn.disabled = false;
        deepScanBtn.innerHTML = originalBtnText;
    }
}


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
        UI.showNotification('Please enter a claim, question, or URL to verify.', 'warning');
        return;
    }

    // Validate input length
    if (input.length > MAX_INPUT_LENGTH) {
        UI.showNotification(`Input too long. Maximum ${MAX_INPUT_LENGTH} characters allowed.`, 'warning');
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
            UI.showNotification('Failed to connect to the fact-check service. Make sure the backend is running on port 5000.', 'error');
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
        UI.showNotification(result.explanation || 'Could not verify this claim.', 'error');
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

    // Update claim title only (Summary removed as per request)
    const claimTitle = document.getElementById('claimTitle');
    if (claimTitle) claimTitle.textContent = "Verification Results";

    // Show feedback buttons
    const feedbackButtons = document.getElementById('feedbackButtons');
    if (feedbackButtons) {
        feedbackButtons.classList.remove('hidden');
        feedbackButtons.classList.add('flex');
    }

    // Update deep scan description with temporal context if available
    if (result.temporal_context && result.temporal_context.description) {
        const deepScanDescElement = document.getElementById('deepScanDescription');
        if (deepScanDescElement) {
            deepScanDescElement.textContent = `Run a deeper scan including cross-referencing ${result.temporal_context.description}.`;
        }
    }

    // Enable deep scan button after regular check
    const deepScanBtn = document.getElementById('deepScanBtn');
    if (deepScanBtn) {
        deepScanBtn.disabled = false;
    }

    // Enable PDF export button
    const exportPdfBtn = document.getElementById('exportPdfBtn');
    if (exportPdfBtn) {
        exportPdfBtn.disabled = false;
    }

    // Render content for the current tab
    renderTabContent(result, currentTab);


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
        case 'reasoning':
            sourcesContainer.innerHTML = buildReasoningHTML(result);
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
 * Build AI Reasoning tab content - shows source stances and confidence breakdown
 */
function buildReasoningHTML(result) {
    const sourceAnalysis = result.source_analysis || [];
    const confidenceBreakdown = result.confidence_breakdown || {};

    // Build confidence breakdown visualization
    const confidenceHTML = `
        <div class="space-y-3">
            <h4 class="text-white font-medium text-sm flex items-center gap-2">
                <span class="material-symbols-outlined text-primary">analytics</span>
                Confidence Breakdown
            </h4>
            <p class="text-slate-400 text-xs mb-3">${escapeHTML(confidenceBreakdown.explanation || 'How we calculated the confidence score.')}</p>
            
            <div class="space-y-2">
                ${buildConfidenceBar('Source Quality', confidenceBreakdown.source_quality || 0, 25, 'High-trust sources increase confidence')}
                ${buildConfidenceBar('Source Quantity', confidenceBreakdown.source_quantity || 0, 20, 'More sources provide better coverage')}
                ${buildConfidenceBar('Fact-Check Sites', confidenceBreakdown.factcheck_found || 0, 25, 'Professional fact-checkers found')}
                ${buildConfidenceBar('Source Consensus', confidenceBreakdown.consensus || 0, 30, 'Agreement between sources')}
            </div>
            
            <div class="mt-4 p-3 rounded-lg bg-primary/10 border border-primary/20">
                <div class="flex items-center justify-between">
                    <span class="text-sm text-slate-300">Total Confidence</span>
                    <span class="text-2xl font-bold text-primary">${result.confidence}%</span>
                </div>
            </div>
        </div>
    `;

    // Build source stance analysis
    const sourceStancesHTML = sourceAnalysis.length > 0 ? `
        <div class="space-y-3 mt-6">
            <h4 class="text-white font-medium text-sm flex items-center gap-2">
                <span class="material-symbols-outlined text-primary">balance</span>
                Source Stance Analysis
            </h4>
            <p class="text-slate-400 text-xs mb-3">How each source positions relative to the claim</p>
            
            <div class="space-y-3">
                ${sourceAnalysis.map(source => {
        const stanceConfig = getStanceConfig(source.stance);
        return `
                        <div class="bg-background-dark/50 rounded-lg p-4 border border-white/5">
                            <div class="flex items-start justify-between gap-3 mb-2">
                                <h5 class="text-white font-medium text-sm flex-1">${escapeHTML(source.source_title || 'Unknown Source')}</h5>
                                <span class="px-2 py-1 rounded-md ${stanceConfig.bgClass} ${stanceConfig.textClass} text-xs font-bold border ${stanceConfig.borderClass} shrink-0">
                                    ${source.stance || 'NEUTRAL'}
                                </span>
                            </div>
                            <div class="flex items-center gap-2 mb-2">
                                <span class="text-xs text-slate-500">Relevance:</span>
                                <div class="flex-1 h-2 bg-slate-700 rounded-full overflow-hidden">
                                    <div class="h-full bg-primary rounded-full transition-all" style="width: ${source.relevance || 50}%"></div>
                                </div>
                                <span class="text-xs text-slate-400 font-mono">${source.relevance || 50}%</span>
                            </div>
                            ${source.key_excerpt ? `
                                <div class="mt-3 p-3 rounded-lg bg-card-dark border border-white/5">
                                    <p class="text-xs text-slate-400 mb-1">Key Excerpt:</p>
                                    <p class="text-sm text-slate-300 italic">"${escapeHTML(source.key_excerpt)}"</p>
                                </div>
                            ` : ''}
                        </div>
                    `;
    }).join('')}
            </div>
        </div>
    ` : `
        <div class="mt-6 flex flex-col items-center justify-center py-8 text-center">
            <div class="size-12 rounded-full bg-slate-500/10 flex items-center justify-center mb-3">
                <span class="material-symbols-outlined text-2xl text-slate-400">analytics</span>
            </div>
            <h4 class="text-white font-medium mb-1">No Source Analysis Available</h4>
            <p class="text-slate-400 text-sm">Detailed source stance analysis was not generated for this check.</p>
        </div>
    `;

    // Contradictions found section
    const contradictionsHTML = result.contradictions_found ? `
        <div class="mt-6 p-4 rounded-lg bg-warning/10 border border-warning/20">
            <div class="flex items-center gap-2 text-warning mb-2">
                <span class="material-symbols-outlined">warning</span>
                <h4 class="font-bold text-sm">Contradictions Detected</h4>
            </div>
            <p class="text-sm text-slate-300">Our analysis found conflicting information between sources. Review the source stances above to understand different perspectives.</p>
        </div>
    ` : '';

    return `
        <div class="space-y-6">
            ${confidenceHTML}
            ${contradictionsHTML}
            ${sourceStancesHTML}
        </div>
    `;
}

/**
 * Build a confidence breakdown bar
 */
function buildConfidenceBar(label, value, maxValue, tooltip) {
    const percentage = (value / maxValue) * 100;
    return `
        <div class="group relative">
            <div class="flex items-center justify-between mb-1">
                <span class="text-xs text-slate-400 flex items-center gap-1">
                    ${label}
                    <span class="material-symbols-outlined text-[14px] text-slate-500 cursor-help" title="${tooltip}">info</span>
                </span>
                <span class="text-xs text-white font-mono">${value}/${maxValue}</span>
            </div>
            <div class="h-2 bg-slate-700 rounded-full overflow-hidden">
                <div class="h-full bg-gradient-to-r from-primary to-accent-success rounded-full transition-all duration-500" 
                     style="width: ${percentage}%"></div>
            </div>
            <div class="hidden group-hover:block absolute left-0 -top-8 bg-card-dark border border-white/10 rounded-lg px-2 py-1 text-xs text-slate-300 whitespace-nowrap z-10">
                ${tooltip}
            </div>
        </div>
    `;
}

/**
 * Get styling config for source stance
 */
function getStanceConfig(stance) {
    const configs = {
        'SUPPORTS': { bgClass: 'bg-success/20', textClass: 'text-success', borderClass: 'border-success/30' },
        'REFUTES': { bgClass: 'bg-danger/20', textClass: 'text-danger', borderClass: 'border-danger/30' },
        'NEUTRAL': { bgClass: 'bg-slate-500/20', textClass: 'text-slate-400', borderClass: 'border-slate-500/30' },
        'MIXED': { bgClass: 'bg-warning/20', textClass: 'text-warning', borderClass: 'border-warning/30' }
    };
    return configs[stance] || configs['NEUTRAL'];
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
        // Check for Verify button by looking for verify icon or loading state
        const isVerifyBtn = btn.textContent.includes('Verify') ||
            btn.textContent.includes('Analyzing') ||
            btn.textContent.includes('Classifying') ||
            btn.textContent.includes('Searching') ||
            btn.textContent.includes('Building') ||
            btn.textContent.includes('Initializing') ||
            (btn.disabled && btn.querySelector('.animate-spin'));

        if (isVerifyBtn) {
            btn.disabled = loading;
            btn.innerHTML = loading
                ? '<span class="material-symbols-outlined text-[20px] animate-spin">progress_activity</span> Analyzing...'
                : '<span class="material-symbols-outlined text-[20px]">verified</span> Verify Credibility';
        }
    });
}


/**
 * Export the current fact-check result as PDF
 */
function exportToPDF() {
    if (!currentResult) {
        UI.showNotification('No fact-check result to export', 'warning');
        return;
    }

    try {
        const { jsPDF } = window.jspdf;
        const doc = new jsPDF();

        let yPos = 20;
        const pageWidth = doc.internal.pageSize.getWidth();
        const margin = 20;
        const maxWidth = pageWidth - 2 * margin;

        // Title
        doc.setFontSize(20);
        doc.setFont(undefined, 'bold');
        doc.text('VisioNova Fact-Check Report', margin, yPos);
        yPos += 10;

        // Verdict Badge
        doc.setFontSize(16);
        const verdictColor = {
            'TRUE': [0, 217, 145],
            'FALSE': [255, 74, 74],
            'PARTIALLY TRUE': [255, 183, 74],
            'MISLEADING': [255, 183, 74],
            'UNVERIFIABLE': [148, 163, 184]
        }[currentResult.verdict] || [148, 163, 184];

        doc.setTextColor(...verdictColor);
        doc.text(currentResult.verdict || 'UNKNOWN', margin, yPos);
        doc.setTextColor(0, 0, 0);
        yPos += 8;

        // Confidence Score
        doc.setFontSize(12);
        doc.text(`Confidence: ${currentResult.confidence}%`, margin, yPos);
        yPos += 10;

        // Separator
        doc.setDrawColor(200, 200, 200);
        doc.line(margin, yPos, pageWidth - margin, yPos);
        yPos += 10;

        // Claim
        doc.setFontSize(14);
        doc.setFont(undefined, 'bold');
        doc.text('Claim:', margin, yPos);
        yPos += 7;

        doc.setFont(undefined, 'normal');
        doc.setFontSize(11);
        const claimText = currentResult.claim || currentResult.url || 'N/A';
        const splitClaim = doc.splitTextToSize(claimText, maxWidth);
        doc.text(splitClaim, margin, yPos);
        yPos += (splitClaim.length * 6) + 8;

        // Summary
        if (currentResult.summary && currentResult.summary.one_liner) {
            doc.setFontSize(14);
            doc.setFont(undefined, 'bold');
            doc.text('Summary:', margin, yPos);
            yPos += 7;

            doc.setFont(undefined, 'normal');
            doc.setFontSize(11);
            const summaryText = doc.splitTextToSize(currentResult.summary.one_liner, maxWidth);
            doc.text(summaryText, margin, yPos);
            yPos += (summaryText.length * 6) + 8;
        }

        // Key Points
        if (currentResult.summary && currentResult.summary.key_points && currentResult.summary.key_points.length > 0) {
            doc.setFontSize(14);
            doc.setFont(undefined, 'bold');
            doc.text('Key Points:', margin, yPos);
            yPos += 7;

            doc.setFont(undefined, 'normal');
            doc.setFontSize(10);

            currentResult.summary.key_points.forEach((point, index) => {
                if (yPos > 270) {
                    doc.addPage();
                    yPos = 20;
                }
                const bulletPoint = `• ${point}`;
                const splitPoint = doc.splitTextToSize(bulletPoint, maxWidth - 5);
                doc.text(splitPoint, margin + 2, yPos);
                yPos += (splitPoint.length * 5) + 3;
            });
            yPos += 5;
        }

        // Sources
        if (currentResult.sources && currentResult.sources.length > 0) {
            if (yPos > 240) {
                doc.addPage();
                yPos = 20;
            }

            doc.setFontSize(14);
            doc.setFont(undefined, 'bold');
            doc.text(`Sources (${currentResult.sources.length}):`, margin, yPos);
            yPos += 7;

            doc.setFont(undefined, 'normal');
            doc.setFontSize(9);

            currentResult.sources.slice(0, 10).forEach((source, index) => {
                if (yPos > 270) {
                    doc.addPage();
                    yPos = 20;
                }

                const sourceNum = `[${index + 1}] `;
                const sourceTitle = source.title || source.domain || 'Source';
                const trustLabel = source.is_factcheck ? 'Fact-Check' : source.trust_level;

                doc.setFont(undefined, 'bold');
                doc.text(sourceNum, margin, yPos);
                doc.setFont(undefined, 'normal');

                const titleSplit = doc.splitTextToSize(sourceTitle, maxWidth - 15);
                doc.text(titleSplit, margin + 8, yPos);
                yPos += (titleSplit.length * 4) + 2;

                doc.setTextColor(100, 100, 100);
                doc.text(`${source.domain} • ${trustLabel}`, margin + 8, yPos);
                doc.setTextColor(0, 0, 0);
                yPos += 4;

                doc.setFontSize(8);
                doc.setTextColor(50, 50, 200);
                const urlSplit = doc.splitTextToSize(source.url, maxWidth - 8);
                doc.text(urlSplit, margin + 8, yPos);
                doc.setTextColor(0, 0, 0);
                doc.setFontSize(9);
                yPos += (urlSplit.length * 3) + 5;
            });
        }

        // Footer
        const pageCount = doc.internal.getNumberOfPages();
        for (let i = 1; i <= pageCount; i++) {
            doc.setPage(i);
            doc.setFontSize(8);
            doc.setTextColor(150, 150, 150);
            doc.text(`Generated by VisioNova | ${new Date().toLocaleDateString()} | Page ${i} of ${pageCount}`,
                margin, doc.internal.pageSize.getHeight() - 10);
        }

        // Save PDF
        const filename = `VisioNova_FactCheck_${Date.now()}.pdf`;
        doc.save(filename);

        UI.showNotification('PDF exported successfully!', 'success');

    } catch (error) {
        console.error('[PDF Export] Error:', error);
        UI.showNotification('Failed to export PDF. Please try again.', 'error');
    }
}

// Initialize when DOM is loaded
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initFactCheck);
} else {
    initFactCheck();
}

