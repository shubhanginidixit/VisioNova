/**
 * VisioNova Text Result Page - Dynamic Analysis
 * Performs client-side text analysis and displays results
 */

const API_BASE_URL = 'http://localhost:5000';

// Analysis state
let textData = null;
let analysisResults = null;

/**
 * Initialize the page
 */
document.addEventListener('DOMContentLoaded', async function () {
    console.log('[TextResult] Page loaded');
    console.log('[TextResult] Checking for data...');

    textData = VisioNovaStorage.getFile('text');
    console.log('[TextResult] textData from storage:', textData);

    // Check for cached result from dashboard
    const cachedResult = sessionStorage.getItem('visioNova_text_result');
    console.log('[TextResult] Cached result exists:', !!cachedResult);

    let preloadedResult = null;
    if (cachedResult) {
        try {
            preloadedResult = JSON.parse(cachedResult);
            sessionStorage.removeItem('visioNova_text_result');
            console.log('[TextResult] Using cached result from dashboard');
            console.log('[TextResult] Result keys:', Object.keys(preloadedResult));
        } catch (e) {
            console.warn('[TextResult] Failed to parse cached result');
        }
    }

    // Check if this was a document upload (PDF/DOCX)
    const isDocument = sessionStorage.getItem('visioNova_isDocument') === 'true';
    const documentFileName = sessionStorage.getItem('visioNova_documentFileName');
    console.log('[TextResult] isDocument:', isDocument);
    console.log('[TextResult] documentFileName:', documentFileName);

    // For documents, we don't have textData from storage (file was sent directly)
    if (isDocument && preloadedResult) {
        console.log('[TextResult] Processing document result');

        // Document was processed by backend
        sessionStorage.removeItem('visioNova_isDocument');
        sessionStorage.removeItem('visioNova_documentFileName');

        updateElement('pageTitle', 'Analysis: ' + (documentFileName || 'Document'));
        updateElement('analysisDate', formatAnalysisDate(new Date()));

        // Show document info
        let displayTextContent = `[Document: ${documentFileName || 'Uploaded Document'}]\n\n`;
        if (preloadedResult.file_info) {
            const info = preloadedResult.file_info;
            displayTextContent += `Format: ${info.format?.toUpperCase() || 'PDF'}\n`;
            displayTextContent += `Pages: ${info.pages || 'N/A'}\n`;
            displayTextContent += `Characters: ${(info.char_count || 0).toLocaleString()}\n\n`;
        }
        displayTextContent += 'Document text extracted and analyzed successfully.';

        console.log('[TextResult] Displaying document info');
        displayText(displayTextContent);
        await analyzeText('', preloadedResult);  // Pass empty text, use preloaded result

    } else if (textData && textData.data) {
        console.log('[TextResult] Processing regular text');

        // Regular text input
        updateElement('pageTitle', 'Analysis: ' + (textData.fileName || 'Text'));
        updateElement('analysisDate', formatAnalysisDate(new Date()));

        displayText(textData.data);
        await analyzeText(textData.data, preloadedResult);
        VisioNovaStorage.clearFile('text');
    } else {
        console.warn('[TextResult] No data found!');
        console.log('[TextResult] textData:', textData);
        console.log('[TextResult] preloadedResult:', preloadedResult);
        console.log('[TextResult] isDocument:', isDocument);

        updateElement('pageTitle', 'Text Analysis');
        showNoTextState();
    }
});

/**
 * Display the text content
 */
function displayText(text) {
    const placeholder = document.getElementById('noTextPlaceholder');
    const uploadedText = document.getElementById('uploadedText');

    if (uploadedText) {
        const paragraphs = text.split('\n\n').filter(p => p.trim());
        let formattedHtml = paragraphs.map(p =>
            `<p class="mb-4">${escapeHtml(p).replace(/\n/g, '<br>')}</p>`
        ).join('');

        if (!formattedHtml) {
            formattedHtml = `<p class="mb-4">${escapeHtml(text).replace(/\n/g, '<br>')}</p>`;
        }

        uploadedText.innerHTML = formattedHtml;
        uploadedText.classList.remove('hidden');
    }
    if (placeholder) {
        placeholder.classList.add('hidden');
    }
}

/**
 * Show no text state
 */
function showNoTextState() {
    updateElement('credibilityScore', '--');
    updateElement('verdictBadge', 'No Text');
    updateElement('sourceLevel', 'N/A');
    updateElement('sourceInfo', 'No text to analyze');
}

/**
 * Main analysis function - calls backend API first, falls back to client-side
 * @param {string} text - The text to analyze
 * @param {object} preloadedResult - Optional preloaded result from dashboard
 */
async function analyzeText(text, preloadedResult = null) {
    // Detect if this is a document file (PDF/DOCX) - skip client-side metrics for binary data
    const isDocument = text.startsWith('data:application/') || (preloadedResult && !!preloadedResult.file_info);

    // Calculate local metrics (skip for documents - use backend metrics instead)
    let metrics, claims, sources;
    if (isDocument && preloadedResult && preloadedResult.metrics) {
        // Use backend metrics for documents
        metrics = {
            wordCount: preloadedResult.metrics.word_count || 0,
            charCount: preloadedResult.file_info?.char_count || 0,
            sentenceCount: preloadedResult.metrics.sentence_count || 0,
            perplexity: preloadedResult.metrics.perplexity?.average || 50,
            burstiness: preloadedResult.metrics.burstiness?.score || 0.5
        };
        claims = [];
        sources = analyzeSourceReliability('');
    } else if (!isDocument) {
        metrics = calculateTextMetrics(text);
        claims = extractClaims(text);
        sources = analyzeSourceReliability(text);
    } else {
        // Document without preloaded result - minimal metrics
        metrics = { wordCount: 0, charCount: 0, sentenceCount: 0, perplexity: 50, burstiness: 0.5 };
        claims = [];
        sources = analyzeSourceReliability('');
    }

    // Try to use preloaded result from dashboard first
    let aiProbability = null;
    let backendMetrics = null;

    if (preloadedResult && preloadedResult.success) {
        // Use cached ML model results from dashboard
        aiProbability = {
            ai: preloadedResult.scores.ai_generated,
            human: preloadedResult.scores.human,
            isLikelyAI: preloadedResult.prediction === 'ai_generated',
            confidence: preloadedResult.confidence / 100,
            source: 'ml_model'
        };
        backendMetrics = preloadedResult.metrics || null;
        console.log('[TextResult] Using preloaded ML result:', preloadedResult.prediction);
    } else {
        // Try backend ML model
        try {
            showLoadingState();
            const backendResult = await callTextDetectionAPI(text);

            if (backendResult && backendResult.success) {
                aiProbability = {
                    ai: backendResult.scores.ai_generated,
                    human: backendResult.scores.human,
                    isLikelyAI: backendResult.prediction === 'ai_generated',
                    confidence: backendResult.confidence / 100,
                    source: 'ml_model'
                };
                backendMetrics = backendResult.metrics || null;
                console.log('[TextResult] Using ML model prediction:', backendResult.prediction);
            }
        } catch (error) {
            console.warn('[TextResult] Backend unavailable, using client-side analysis:', error.message);
        }
    }

    // Fall back to client-side heuristics if no ML result
    if (!aiProbability) {
        aiProbability = calculateAIProbability(text, metrics);
        aiProbability.source = 'heuristics';
        console.log('[TextResult] Using client-side heuristics');
    }

    // Merge backend metrics with local metrics if available
    if (backendMetrics) {
        metrics.perplexity = backendMetrics.perplexity?.average || metrics.perplexity;
        metrics.perplexityFlow = backendMetrics.perplexity?.flow;
        metrics.burstiness = backendMetrics.burstiness?.score || metrics.burstiness;
        metrics.burstinenessData = backendMetrics.burstiness?.bars;
        metrics.rhythm = backendMetrics.rhythm;
        metrics.wordCount = backendMetrics.word_count || metrics.wordCount;
        metrics.ngramUniformity = backendMetrics.ngram_uniformity;
    }

    // Store results including any explanation
    analysisResults = {
        metrics,
        aiProbability,
        claims,
        sources,
        sentenceAnalysis: preloadedResult?.sentence_analysis || backendResult?.sentence_analysis || [],
        detectedPatterns: preloadedResult?.detected_patterns || backendResult?.detected_patterns || {},
        explanation: preloadedResult?.explanation || backendResult?.explanation || null
    };

    // Update UI with animations
    setTimeout(() => updateUI(analysisResults), 300);
}

/**
 * Call the backend text detection API
 * @param {string} text - Text to analyze
 * @param {boolean} explain - Request Groq explanation
 */
async function callTextDetectionAPI(text, explain = true) {
    const response = await fetch(`${API_BASE_URL}/api/detect-ai`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: text, explain: explain })
    });

    if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
    }

    return await response.json();
}

/**
 * Show loading state while waiting for API
 */
function showLoadingState() {
    updateElement('credibilityScore', '...');
    updateElement('verdictBadge', 'Analyzing');
    const verdictBadge = document.getElementById('verdictBadge');
    if (verdictBadge) {
        verdictBadge.className = 'px-2 py-0.5 rounded-full bg-primary/20 text-primary text-xs font-bold border border-primary/20 animate-pulse';
    }
}

/**
 * Calculate text metrics (perplexity, burstiness, etc.)
 */
function calculateTextMetrics(text) {
    const words = text.split(/\s+/).filter(w => w);
    const sentences = text.split(/[.!?]+/).filter(s => s.trim());
    const paragraphs = text.split(/\n\n+/).filter(p => p.trim());

    // Word statistics
    const wordCount = words.length;
    const charCount = text.length;
    const avgWordLength = words.reduce((sum, w) => sum + w.length, 0) / wordCount || 0;

    // Sentence statistics
    const sentenceCount = sentences.length;
    const wordsPerSentence = sentences.map(s => s.split(/\s+/).filter(w => w).length);
    const avgWordsPerSentence = wordsPerSentence.reduce((a, b) => a + b, 0) / sentenceCount || 0;

    // Burstiness - variance in sentence length (high = human, low = AI)
    const sentenceVariance = calculateVariance(wordsPerSentence);
    const burstiness = Math.min(1, sentenceVariance / 50); // Normalize to 0-1

    // Vocabulary richness (unique words / total words)
    const uniqueWords = new Set(words.map(w => w.toLowerCase()));
    const vocabularyRichness = uniqueWords.size / wordCount || 0;

    // Simulated perplexity (based on text characteristics)
    // Real perplexity would require a language model
    const basePerplexity = 30 + (vocabularyRichness * 40) + (burstiness * 30);
    const perplexity = Math.min(100, Math.max(10, basePerplexity + (Math.random() * 10 - 5)));

    return {
        wordCount,
        charCount,
        sentenceCount,
        paragraphCount: paragraphs.length,
        avgWordLength: avgWordLength.toFixed(1),
        avgWordsPerSentence: avgWordsPerSentence.toFixed(1),
        burstiness: burstiness.toFixed(2),
        vocabularyRichness: (vocabularyRichness * 100).toFixed(1),
        perplexity: perplexity.toFixed(1)
    };
}

/**
 * Calculate AI probability based on text patterns
 */
function calculateAIProbability(text, metrics) {
    let aiScore = 50; // Start neutral

    // AI text tends to have:
    // - Lower burstiness (more uniform sentences)
    if (parseFloat(metrics.burstiness) < 0.3) aiScore += 15;
    else if (parseFloat(metrics.burstiness) > 0.6) aiScore -= 15;

    // - Lower vocabulary richness (more repetitive)
    if (parseFloat(metrics.vocabularyRichness) < 40) aiScore += 10;
    else if (parseFloat(metrics.vocabularyRichness) > 60) aiScore -= 10;

    // - Predictable sentence lengths
    if (parseFloat(metrics.avgWordsPerSentence) > 15 && parseFloat(metrics.avgWordsPerSentence) < 25) {
        aiScore += 5; // AI tends toward medium-length sentences
    }

    // - Certain phrase patterns (simplified check)
    const aiPatterns = [
        /as an ai/i, /language model/i, /i cannot/i, /i don't have/i,
        /it'?s important to note/i, /in conclusion/i, /on the other hand/i,
        /furthermore/i, /moreover/i, /in summary/i, /delve into/i
    ];
    const patternMatches = aiPatterns.filter(p => p.test(text)).length;
    aiScore += patternMatches * 8;

    // Bound the score
    aiScore = Math.max(5, Math.min(95, aiScore));

    return {
        ai: aiScore,
        human: 100 - aiScore,
        isLikelyAI: aiScore > 50,
        confidence: Math.abs(aiScore - 50) / 50 // 0-1 confidence
    };
}

/**
 * Extract potential claims from text
 */
function extractClaims(text) {
    const claims = [];
    const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 20);

    // Look for claim-like patterns
    const claimPatterns = [
        /(\d+%|\d+ percent)/i,  // Statistics
        /(study|research|survey|report)\s+(shows?|found|indicates?)/i,
        /(according to|stated that|claimed that)/i,
        /(is|are|was|were)\s+(the|a)\s+(first|largest|smallest|best|worst)/i,
        /\b(in \d{4}|since \d{4}|by \d{4})\b/i  // Dates
    ];

    sentences.forEach(sentence => {
        const trimmed = sentence.trim();
        claimPatterns.forEach((pattern, idx) => {
            if (pattern.test(trimmed) && claims.length < 5) {
                const existing = claims.find(c => c.statement === trimmed);
                if (!existing) {
                    claims.push({
                        statement: trimmed,
                        type: getClaimType(idx),
                        verdict: getRandomVerdict(),
                        source: getRandomSource()
                    });
                }
            }
        });
    });

    return claims.slice(0, 4); // Max 4 claims
}

/**
 * Analyze source reliability indicators in text
 */
function analyzeSourceReliability(text) {
    const domains = [];
    const academicDomains = ['edu', 'gov', 'org', 'ac.uk'];
    const newsDomains = ['reuters', 'bbc', 'nytimes', 'guardian'];

    // Look for URLs or domain mentions
    const urlPattern = /(?:https?:\/\/)?(?:www\.)?([a-z0-9-]+(?:\.[a-z]{2,})+)/gi;
    let match;
    while ((match = urlPattern.exec(text)) !== null) {
        domains.push(match[1]);
    }

    // Look for citation patterns
    const citationPatterns = [
        /according to ([A-Z][a-z]+ [A-Z][a-z]+)/g,
        /\(([A-Z][a-z]+,? \d{4})\)/g,
        /([A-Z][a-z]+ et al\.?,? \d{4})/g
    ];

    const citations = [];
    citationPatterns.forEach(pattern => {
        let m;
        while ((m = pattern.exec(text)) !== null) {
            citations.push(m[1]);
        }
    });

    // Determine reliability level
    let level = 'Unknown';
    let info = 'No verifiable sources found';

    if (domains.length > 0 || citations.length > 0) {
        const hasAcademic = domains.some(d => academicDomains.some(a => d.includes(a)));
        const hasNews = domains.some(d => newsDomains.some(n => d.includes(n)));

        if (hasAcademic) {
            level = 'High Authority';
            info = `Found ${domains.length + citations.length} academic/trusted sources`;
        } else if (hasNews) {
            level = 'Medium Authority';
            info = `Found ${domains.length} news sources`;
        } else if (domains.length > 0) {
            level = 'Low Authority';
            info = `Found ${domains.length} web sources`;
        } else if (citations.length > 0) {
            level = 'Medium Authority';
            info = `Found ${citations.length} citation(s)`;
        }
    }

    return { level, info, domains: domains.slice(0, 3), citations: citations.slice(0, 3) };
}

/**
 * Update all UI elements with analysis results
 */
function updateUI(results) {
    const { metrics, aiProbability, claims, sources } = results;

    // Calculate credibility score (inverted AI probability)
    const credibilityScore = aiProbability.human;

    // Update credibility scorecard
    updateElement('credibilityScore', `${Math.round(credibilityScore)}%`);
    updateElement('scoreBar', null, { width: `${credibilityScore}%` });
    updateElement('metricsInfo', `Based on ${metrics.wordCount} words analyzed`);

    // Update verdict badge
    const verdictBadge = document.getElementById('verdictBadge');
    if (verdictBadge) {
        if (credibilityScore >= 70) {
            verdictBadge.textContent = 'Likely Human';
            verdictBadge.className = 'px-2 py-0.5 rounded-full bg-accent-emerald/20 text-accent-emerald text-xs font-bold border border-accent-emerald/20';
        } else if (credibilityScore >= 40) {
            verdictBadge.textContent = 'Uncertain';
            verdictBadge.className = 'px-2 py-0.5 rounded-full bg-accent-amber/20 text-accent-amber text-xs font-bold border border-accent-amber/20';
        } else {
            verdictBadge.textContent = 'Likely AI';
            verdictBadge.className = 'px-2 py-0.5 rounded-full bg-accent-red/20 text-accent-red text-xs font-bold border border-accent-red/20';
        }
    }

    // Update probability bars
    updateElement('humanBar', null, { height: `${aiProbability.human}%` });
    updateElement('humanPercent', `${Math.round(aiProbability.human)}%`);
    updateElement('aiBar', null, { height: `${aiProbability.ai}%` });
    updateElement('aiPercent', `${Math.round(aiProbability.ai)}%`);

    // Update probability note
    if (aiProbability.human >= 70) {
        updateElement('probabilityText', 'Consistent human writing patterns');
    } else if (aiProbability.ai >= 70) {
        updateElement('probabilityText', 'AI-generated patterns detected');
        const note = document.getElementById('probabilityNote');
        if (note) note.className = note.className.replace('text-accent-emerald', 'text-accent-red');
    } else {
        updateElement('probabilityText', 'Mixed writing patterns detected');
        const note = document.getElementById('probabilityNote');
        if (note) note.className = note.className.replace('text-accent-emerald', 'text-accent-amber');
    }

    // Update source reliability
    updateElement('sourceLevel', sources.level);
    updateElement('sourceInfo', sources.info);
    updateSourceTags(sources.domains);

    // Update perplexity chart
    updateElement('perplexityAvg', `Avg: ${metrics.perplexity}`);
    if (metrics.perplexityFlow) {
        updatePerplexityChart(metrics.perplexityFlow);
    }

    // Update burstiness chart
    if (metrics.burstinenessData) {
        updateBurstinessChart(metrics.burstinenessData);
    }

    // Update rhythm uniformity
    if (metrics.rhythm) {
        updateRhythmStatus(metrics.rhythm);
    }
}

/**
 * Update source tags display
 */
function updateSourceTags(domains) {
    const container = document.getElementById('sourceTags');
    if (!container) return;

    if (domains.length === 0) {
        container.innerHTML = '<span class="px-2 py-1 bg-white/5 rounded text-xs text-white/50 border border-white/5">No sources detected</span>';
        return;
    }

    container.innerHTML = domains.map(d =>
        `<span class="px-2 py-1 bg-white/5 rounded text-xs text-white/70 border border-white/5">${escapeHtml(d)}</span>`
    ).join('');
}

/**
 * Update claims table with extracted claims
 */
function updateClaimsTable(claims) {
    const tbody = document.getElementById('claimsTableBody');
    if (!tbody) return;

    if (claims.length === 0) {
        tbody.innerHTML = `
            <tr class="border-b border-white/5">
                <td colspan="4" class="p-8 text-center text-white/50">
                    <span class="material-symbols-outlined text-4xl mb-2 block opacity-30">fact_check</span>
                    No verifiable claims detected in this text
                </td>
            </tr>
        `;
        return;
    }

    tbody.innerHTML = claims.map((claim, idx) => `
        <tr class="${idx < claims.length - 1 ? 'border-b border-white/5' : ''} hover:bg-white/5 transition-colors">
            <td class="p-4 text-white font-medium">"${escapeHtml(claim.statement.substring(0, 100))}${claim.statement.length > 100 ? '...' : ''}"</td>
            <td class="p-4">${getVerdictBadge(claim.verdict)}</td>
            <td class="p-4">${claim.source ? getSourceLink(claim.source) : '<span class="text-white/50 italic">No source found</span>'}</td>
            <td class="p-4 text-right">
                <button class="text-white/40 hover:text-white transition-colors">
                    <span class="material-symbols-outlined">more_vert</span>
                </button>
            </td>
        </tr>
    `).join('');
}

// ============= Helper Functions =============

function updateElement(id, text, styles = null) {
    const el = document.getElementById(id);
    if (!el) return;
    if (text !== null) el.textContent = text;
    if (styles) Object.assign(el.style, styles);
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatAnalysisDate(date) {
    return `Analyzed on ${date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })} • ${date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })}`;
}

function calculateVariance(arr) {
    if (arr.length === 0) return 0;
    const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
    return arr.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / arr.length;
}

function getClaimType(idx) {
    const types = ['statistic', 'research', 'quote', 'superlative', 'date'];
    return types[idx] || 'general';
}

function getRandomVerdict() {
    const verdicts = ['Verified', 'Verified', 'Unverified', 'False'];
    return verdicts[Math.floor(Math.random() * verdicts.length)];
}

function getRandomSource() {
    const sources = [
        { name: 'Wikipedia', initial: 'W' },
        { name: 'Nature Journal', initial: 'N' },
        { name: 'MIT Tech Review', initial: 'M' },
        { name: 'Reuters', initial: 'R' },
        null // No source
    ];
    return sources[Math.floor(Math.random() * sources.length)];
}

function getVerdictBadge(verdict) {
    const configs = {
        'Verified': { icon: 'check_circle', bg: 'bg-accent-emerald/10', text: 'text-accent-emerald', border: 'border-accent-emerald/20' },
        'Unverified': { icon: 'warning', bg: 'bg-accent-amber/10', text: 'text-accent-amber', border: 'border-accent-amber/20' },
        'False': { icon: 'cancel', bg: 'bg-accent-red/10', text: 'text-accent-red', border: 'border-accent-red/20' }
    };
    const c = configs[verdict] || configs['Unverified'];
    return `<span class="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-bold ${c.bg} ${c.text} border ${c.border}">
        <span class="material-symbols-outlined text-[14px]">${c.icon}</span>${verdict}
    </span>`;
}

function getSourceLink(source) {
    return `<a class="flex items-center gap-2 text-primary hover:text-white transition-colors group" href="#">
        <div class="size-5 rounded bg-white/10 flex items-center justify-center text-[10px] text-white">${source.initial}</div>
        <span class="underline decoration-primary/30 group-hover:decoration-white">${source.name}</span>
        <span class="material-symbols-outlined text-[14px] opacity-0 group-hover:opacity-100">open_in_new</span>
    </a>`;
}

/**
 * Update the perplexity line chart with dynamic data
 * @param {number[]} flowData - Array of perplexity values for each section
 */
function updatePerplexityChart(flowData) {
    const svg = document.querySelector('.bg-primary-dark\\/40 svg');
    if (!svg || !flowData || flowData.length < 2) return;

    // Generate SVG path from flow data
    const width = 360;
    const height = 100;
    const padding = 10;

    const pointWidth = (width - padding * 2) / (flowData.length - 1);
    const minVal = Math.min(...flowData);
    const maxVal = Math.max(...flowData);
    const range = maxVal - minVal || 1;

    // Generate points
    const points = flowData.map((val, i) => {
        const x = padding + i * pointWidth;
        // Invert Y because SVG Y increases downward
        const y = height - padding - ((val - minVal) / range) * (height - padding * 2);
        return { x, y };
    });

    // Create smooth curve path using bezier curves
    let pathD = `M${points[0].x},${points[0].y}`;
    for (let i = 1; i < points.length; i++) {
        const prev = points[i - 1];
        const curr = points[i];
        const cpx = (prev.x + curr.x) / 2;
        pathD += ` C${cpx},${prev.y} ${cpx},${curr.y} ${curr.x},${curr.y}`;
    }

    // Create fill path
    const fillD = `${pathD} V ${height} H ${points[0].x} Z`;

    // Update the paths in SVG
    const paths = svg.querySelectorAll('path');
    if (paths[0]) paths[0].setAttribute('d', pathD);
    if (paths[1]) paths[1].setAttribute('d', fillD);
}

/**
 * Update the burstiness bar chart with dynamic data
 * @param {object} data - Contains document bars and human_baseline bars
 */
function updateBurstinessChart(data) {
    const container = document.querySelector('.bg-primary-dark\\/40 .flex-1.flex.items-end');
    if (!container || !data || !data.document) return;

    const docBars = data.document;
    const humanBars = data.human_baseline || [];

    // Generate bar HTML
    let barsHtml = '';
    for (let i = 0; i < Math.max(docBars.length, 6); i++) {
        const docHeight = docBars[i] || 20;
        const humanHeight = humanBars[i] || 50;

        barsHtml += `
            <div class="w-full bg-white/5 rounded-t relative group transition-all" style="height: ${docHeight}%">
                <div class="absolute bottom-0 w-full bg-accent-emerald/80 rounded-t transition-all" 
                     style="height: ${humanHeight}%; width: 60%; left: 20%"></div>
            </div>
        `;
    }

    container.innerHTML = barsHtml;
}

/**
 * Update the rhythm uniformity status display
 * @param {object} rhythm - Contains status and description
 */
function updateRhythmStatus(rhythm) {
    // Find the rhythm card (last card in the metrics section)
    const rhythmCard = document.querySelector('.bg-primary-dark\\/40.flex.items-center.justify-between');
    if (!rhythmCard || !rhythm) return;

    // Update the description text
    const descEl = rhythmCard.querySelector('p.text-white\\/40');
    if (descEl) {
        descEl.textContent = rhythm.description;
    }

    // Update the status badge
    const badge = rhythmCard.querySelector('span.px-3');
    if (badge) {
        badge.textContent = rhythm.status;

        // Update badge color based on status
        badge.className = 'px-3 py-1 rounded-full text-xs font-bold border ';
        if (rhythm.status === 'Normal') {
            badge.className += 'bg-accent-emerald/10 text-accent-emerald border-accent-emerald/20';
        } else if (rhythm.status === 'Uniform') {
            badge.className += 'bg-accent-amber/10 text-accent-amber border-accent-amber/20';
        } else {
            badge.className += 'bg-white/5 text-white border-white/10';
        }
    }
}
