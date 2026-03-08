/**
 * VisioNova Audio Result Page
 * Handles segmented audio analysis, timeline rendering, and dynamic UI updates.
 *
 * Backend response shape:
 *   { success, prediction, verdict, fake_probability, real_probability, confidence,
 *     ensemble_details[], artifacts_detected[], analysis_mode, total_duration_seconds,
 *     segments_analyzed, segments[{start_sec,end_sec,fake_probability,real_probability,verdict}],
 *     meta:{duration_seconds, sample_rate, segment_length_sec, segment_overlap_sec} }
 */

document.addEventListener('DOMContentLoaded', async function () {
    // Check if the dashboard already performed the analysis and cached the result
    const cachedResult = sessionStorage.getItem('visioNova_audio_result');
    if (cachedResult) {
        sessionStorage.removeItem('visioNova_audio_result');
        try {
            const result = JSON.parse(cachedResult);
            if (result.success) {
                // Set page title from cached result metadata
                setText('pageTitle', result.meta?.fileName || 'Audio Analysis');

                // Try to show audio player if audio data is still available
                let audioData = await VisioNovaStorage.getAudioFile().catch(() => null);
                if (!audioData) audioData = VisioNovaStorage.getFile('audio');
                if (audioData) {
                    const fileBadge = document.getElementById('fileBadge');
                    if (fileBadge && audioData.fileName) {
                        const ext = audioData.fileName.split('.').pop().toUpperCase();
                        fileBadge.textContent = ext;
                        fileBadge.classList.remove('hidden');
                    }
                    setText('pageTitle', audioData.fileName || 'Audio Analysis');
                    showAudioPlayer(audioData);
                }

                renderAll(result);
                return;
            }
        } catch (e) {
            console.error('Failed to parse cached audio result:', e);
        }
    }

    // No cached result — need audio data to call the API directly
    let audioData = await VisioNovaStorage.getAudioFile().catch(() => null);
    if (!audioData) {
        audioData = VisioNovaStorage.getFile('audio');
    }

    if (!audioData) {
        window.location.href = 'AnalysisDashboard.html';
        return;
    }

    // Page title & file badge
    setText('pageTitle', audioData.fileName || 'Audio Analysis');
    const fileBadge = document.getElementById('fileBadge');
    if (fileBadge && audioData.fileName) {
        const ext = audioData.fileName.split('.').pop().toUpperCase();
        fileBadge.textContent = ext;
        fileBadge.classList.remove('hidden');
    }

    // Show audio player
    showAudioPlayer(audioData);

    // Start analysis (direct API call fallback)
    await runAudioAnalysis(audioData);
});

/* ===================================================================
   Audio player
   =================================================================== */

let wavesurfer = null;
let wsRegions = null;

function showAudioPlayer(audioData) {
    const placeholder = document.getElementById('noAudioPlaceholder');
    const wrapper = document.getElementById('audioPlayerWrapper');

    if (placeholder && wrapper) {
        placeholder.classList.add('hidden');
        wrapper.classList.remove('hidden');

        // Destroy existing instance if re-analyzing
        if (wavesurfer) {
            wavesurfer.destroy();
        }

        // Initialize WaveSurfer
        wavesurfer = WaveSurfer.create({
            container: '#waveform',
            waveColor: '#00E5FF',       // Neon cyan glow
            progressColor: '#3A8DFF',   // Deep blue progress
            cursorColor: '#ffffff',
            barWidth: 3,
            barGap: 3,
            barRadius: 3,
            height: 128,
            normalize: true,
            backend: 'WebAudio',        // Ensures accurate peak rendering
        });

        // Initialize Regions plugin for forensic highlighting
        wsRegions = wavesurfer.registerPlugin(WaveSurfer.Regions.create());

        const audioUrl = audioData.blobUrl || audioData.data;
        if (audioUrl) {
            wavesurfer.load(audioUrl);
        }

        // Setup Controls
        const playBtn = document.getElementById('playPauseBtn');
        const volumeSlider = document.getElementById('volumeSlider');
        const currentTimeDisplay = document.getElementById('currentTimeDisplay');
        const durationDisplay = document.getElementById('durationDisplay');

        if (playBtn) {
            // Clone to remove old listeners
            const newPlayBtn = playBtn.cloneNode(true);
            playBtn.parentNode.replaceChild(newPlayBtn, playBtn);
            newPlayBtn.onclick = () => wavesurfer.playPause();
        }

        if (volumeSlider) {
            volumeSlider.oninput = (e) => wavesurfer.setVolume(Number(e.target.value));
        }

        // WaveSurfer Events
        wavesurfer.on('play', () => {
            const icon = document.getElementById('playIcon');
            if (icon) icon.textContent = 'pause';
        });
        wavesurfer.on('pause', () => {
            const icon = document.getElementById('playIcon');
            if (icon) icon.textContent = 'play_arrow';
        });

        wavesurfer.on('ready', () => {
            if (durationDisplay) durationDisplay.textContent = formatDuration(wavesurfer.getDuration());
        });

        wavesurfer.on('timeupdate', (currentTime) => {
            if (currentTimeDisplay) currentTimeDisplay.textContent = formatDuration(currentTime);
        });
    }
}

/* ===================================================================
   API call
   =================================================================== */

async function runAudioAnalysis(audioData) {
    try {
        // Check if the dashboard already performed the analysis
        const cachedResult = sessionStorage.getItem('visioNova_audio_result');
        if (cachedResult) {
            sessionStorage.removeItem('visioNova_audio_result');
            const result = JSON.parse(cachedResult);
            if (result.success) {
                renderAll(result);
                return;
            }
        }

        // Fallback: call API directly (e.g., if user navigated here without dashboard)
        const response = await fetch('/api/detect-audio', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                audio: audioData.data,
                filename: audioData.fileName
            })
        });

        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.error || response.statusText);
        }

        const result = await response.json();
        if (!result.success) throw new Error(result.error || 'Analysis failed');

        renderAll(result);
    } catch (error) {
        console.error('Audio analysis error:', error);
        setText('probabilityText', 'Analysis failed — ' + error.message);
        showErrorFindings(error.message);
    }
}

/* ===================================================================
   Master render
   =================================================================== */

function renderAll(r) {
    updateScorecard(r);
    updateModelCard(r);
    renderKeyFindings(r.artifacts_detected || []);
    renderExplanation(r);
    renderModelDetails(r);

    if (r.analysis_mode === 'segmented' && r.segments && r.segments.length > 1) {
        renderSegmentTimeline(r.segments, r.total_duration_seconds);
        renderSegmentSummary(r.segments);
    }

    setupCollapsiblePanels();
    setupActionButtons(r);

    // Metadata badges
    const dur = r.total_duration_seconds || r.meta?.duration_seconds;
    if (dur) {
        const badge = document.getElementById('durationBadge');
        const text = document.getElementById('durationText');
        if (badge && text) { text.textContent = formatDuration(dur); badge.classList.remove('hidden'); }
    }
    const sr = r.meta?.sample_rate;
    if (sr) {
        const badge = document.getElementById('sampleRateBadge');
        const text = document.getElementById('sampleRateText');
        if (badge && text) { text.textContent = sr + ' Hz'; badge.classList.remove('hidden'); }
    }
}

/* ===================================================================
   Scorecard (Human vs AI bars)
   =================================================================== */

function updateScorecard(r) {
    const fakePct = r.fake_probability ?? 50;
    const realPct = r.real_probability ?? (100 - fakePct);

    // Animate bars
    requestAnimationFrame(() => {
        const humanBar = document.getElementById('humanBar');
        const aiBar = document.getElementById('aiBar');
        if (humanBar) humanBar.style.height = realPct + '%';
        if (aiBar) aiBar.style.height = fakePct + '%';
    });

    setText('humanPercent', Math.round(realPct) + '%');
    setText('aiPercent', Math.round(fakePct) + '%');

    // Note below bars
    const isFake = r.prediction === 'ai_generated';
    const noteEl = document.getElementById('probabilityNote');
    const textEl = document.getElementById('probabilityText');
    if (noteEl && textEl) {
        if (isFake) {
            textEl.textContent = `${Math.round(fakePct)}% probability of AI generation detected`;
            noteEl.classList.remove('text-primary');
            noteEl.classList.add('text-accent-danger');
        } else {
            textEl.textContent = `${Math.round(realPct)}% probability of authentic human audio`;
            noteEl.classList.remove('text-accent-danger');
            noteEl.classList.add('text-primary');
        }
    }
}

/* ===================================================================
   Model / Analysis Mode card
   =================================================================== */

function updateModelCard(r) {
    const mode = r.analysis_mode === 'segmented' ? 'Segmented Analysis' : 'Single-Pass Analysis';
    setText('analysisMode', mode);

    const info = [];
    if (r.analysis_mode === 'segmented') {
        info.push(`${r.segments_analyzed} segments analyzed`);
        info.push(`${r.meta?.segment_length_sec ?? 30}s windows, ${r.meta?.segment_overlap_sec ?? 5}s overlap`);
    } else {
        info.push('Full audio analyzed in one pass');
    }
    setText('analysisModeInfo', info.join(' · '));

    // Tags
    const tagsEl = document.getElementById('modelTags');
    if (!tagsEl) return;
    tagsEl.innerHTML = '';

    const tags = [];
    const model = r.ensemble_details?.[0];
    if (model) tags.push(model.name);
    tags.push(r.prediction === 'ai_generated' ? 'AI-Generated' : 'Authentic');
    if (r.analysis_mode === 'segmented') tags.push('Segmented');
    tags.push(`${Math.round(r.confidence)}% confidence`);

    tags.forEach(t => {
        const span = document.createElement('span');
        span.className = 'text-[10px] uppercase font-bold tracking-wide px-2 py-1 rounded bg-white/5 border border-white/10 text-white/50';
        span.textContent = t;
        tagsEl.appendChild(span);
    });

    // Analysis badge
    const ab = document.getElementById('analysisBadge');
    const abt = document.getElementById('analysisBadgeText');
    if (ab && abt) { abt.textContent = r.analysis_mode === 'segmented' ? 'Segmented' : 'Single Pass'; ab.classList.remove('hidden'); }

    if (r.analysis_mode === 'segmented') {
        const sb = document.getElementById('segmentCountBadge');
        const st = document.getElementById('segmentCountText');
        if (sb && st) { st.textContent = r.segments_analyzed + ' segments'; sb.classList.remove('hidden'); }
    }
}

/* ===================================================================
   Segment Timeline
   =================================================================== */

function renderSegmentTimeline(segments, totalDuration) {
    const section = document.getElementById('timelineSection');
    const bar = document.getElementById('timelineBar');
    const endLabel = document.getElementById('timelineEndLabel');

    if (!section || !bar) return;
    section.classList.remove('hidden');
    bar.innerHTML = '';

    // Clear previous forensic regions
    if (wsRegions) wsRegions.clearRegions();

    if (endLabel) endLabel.textContent = formatDuration(totalDuration);

    segments.forEach((seg, idx) => {
        const duration = seg.end_sec - seg.start_sec;
        const widthPct = (duration / totalDuration) * 100;

        const block = document.createElement('div');
        block.className = 'seg-block rounded-sm';
        block.style.width = widthPct + '%';
        block.style.minWidth = '4px';
        block.style.background = segmentColor(seg.fake_probability);
        block.title = `Segment ${idx + 1}: ${formatDuration(seg.start_sec)} – ${formatDuration(seg.end_sec)} | ${Math.round(seg.fake_probability)}% fake`;
        block.dataset.segIdx = idx;

        block.addEventListener('click', () => {
            showSegmentDetail(seg, idx, segments);
            if (wavesurfer) {
                // Jump to slightly before the segment starts
                wavesurfer.setTime(Math.max(0, seg.start_sec - 0.5));
            }
        });
        bar.appendChild(block);

        // Add Red WaveSurfer Region overlay for high probability segments
        if (wsRegions && seg.fake_probability >= 50) {
            const opacity = seg.fake_probability >= 80 ? 0.6 : 0.3;
            // RGB for #FF4A4A with dynamic opacity
            wsRegions.addRegion({
                start: seg.start_sec,
                end: seg.end_sec,
                color: `rgba(255, 74, 74, ${opacity})`,
                drag: false,
                resize: false
            });
        }
    });
}

function segmentColor(fakePct) {
    if (fakePct >= 65) return '#FF4A4A';   // danger
    if (fakePct >= 40) return '#FFB74A';   // warning
    return '#00D991';                       // success
}

function showSegmentDetail(seg, idx, allSegments) {
    const card = document.getElementById('segmentDetail');
    if (!card) return;
    card.classList.remove('hidden');

    setText('segmentDetailTitle', `Segment ${idx + 1}`);
    setText('segDetailTime', `${formatDuration(seg.start_sec)} – ${formatDuration(seg.end_sec)}`);
    setText('segDetailFake', Math.round(seg.fake_probability) + '%');
    setText('segDetailVerdict', seg.verdict === 'likely_ai' ? 'Likely AI' : 'Likely Human');

    // Update color of verdict
    const verdictEl = document.getElementById('segDetailVerdict');
    if (verdictEl) {
        verdictEl.className = seg.verdict === 'likely_ai'
            ? 'text-sm font-bold text-accent-danger'
            : 'text-sm font-bold text-accent-success';
    }

    const fakeEl = document.getElementById('segDetailFake');
    if (fakeEl) {
        fakeEl.className = seg.fake_probability >= 50
            ? 'text-sm font-bold text-accent-danger'
            : 'text-sm font-bold text-accent-success';
    }

    // Highlight active segment block
    document.querySelectorAll('.seg-block').forEach(b => b.classList.remove('active'));
    const activeBlock = document.querySelector(`.seg-block[data-seg-idx="${idx}"]`);
    if (activeBlock) activeBlock.classList.add('active');

    // Close button
    const closeBtn = document.getElementById('segmentDetailClose');
    if (closeBtn) {
        closeBtn.onclick = () => {
            card.classList.add('hidden');
            document.querySelectorAll('.seg-block').forEach(b => b.classList.remove('active'));
        };
    }
}

/* ===================================================================
   Segment Summary card
   =================================================================== */

function renderSegmentSummary(segments) {
    const card = document.getElementById('segmentSummaryCard');
    const content = document.getElementById('segmentSummaryContent');
    if (!card || !content) return;

    card.classList.remove('hidden');
    content.innerHTML = '';

    segments.forEach((seg, idx) => {
        const pct = Math.round(seg.fake_probability);
        const color = seg.verdict === 'likely_ai' ? 'bg-accent-danger' : 'bg-accent-success';
        const textColor = seg.verdict === 'likely_ai' ? 'text-accent-danger' : 'text-accent-success';
        const label = seg.verdict === 'likely_ai' ? 'AI' : 'Real';

        const row = document.createElement('div');
        row.className = 'flex items-center gap-3 text-sm';
        row.innerHTML = `
            <span class="text-white/40 text-xs w-6 text-right shrink-0">${idx + 1}</span>
            <span class="text-white/60 text-xs w-24 shrink-0">${formatDuration(seg.start_sec)} – ${formatDuration(seg.end_sec)}</span>
            <div class="flex-1 h-1.5 bg-white/5 rounded-full overflow-hidden">
                <div class="${color} h-full rounded-full" style="width: ${pct}%"></div>
            </div>
            <span class="${textColor} text-xs font-bold w-12 text-right">${pct}%</span>
            <span class="text-[10px] font-bold uppercase px-1.5 py-0.5 rounded ${color}/20 ${textColor} w-10 text-center">${label}</span>
        `;
        content.appendChild(row);
    });
}

/* ===================================================================
   Key Findings
   =================================================================== */

function renderKeyFindings(artifacts) {
    const container = document.getElementById('keyFindings');
    const countEl = document.getElementById('findingsCount');
    if (!container) return;

    container.innerHTML = '';

    if (!artifacts.length) {
        artifacts = ['Analysis complete — no specific artifacts reported'];
    }

    if (countEl) countEl.textContent = `${artifacts.length} finding${artifacts.length !== 1 ? 's' : ''} identified`;

    artifacts.forEach(text => {
        const severity = artifactSeverity(text);
        const item = document.createElement('div');
        item.className = 'flex items-start gap-3 p-3 bg-white/5 rounded-xl border border-white/5';
        item.innerHTML = `
            <div class="mt-0.5 p-1 rounded-lg ${severity.bg}">
                <span class="material-symbols-outlined !text-[16px] ${severity.text}">${severity.icon}</span>
            </div>
            <span class="text-white/80 text-sm leading-relaxed">${escapeHtml(text)}</span>
        `;
        container.appendChild(item);
    });
}

function artifactSeverity(text) {
    const lower = text.toLowerCase();
    if (lower.includes('vocoder') || lower.includes('synthetic phase') || lower.includes('missing biological'))
        return { icon: 'error', bg: 'bg-accent-danger/10', text: 'text-accent-danger' };
    if (lower.includes('anomal') || lower.includes('unnatural') || lower.includes('spectral'))
        return { icon: 'warning', bg: 'bg-accent-warning/10', text: 'text-accent-warning' };
    if (lower.includes('natural') || lower.includes('confirmed') || lower.includes('biological markers'))
        return { icon: 'check_circle', bg: 'bg-accent-success/10', text: 'text-accent-success' };
    return { icon: 'info', bg: 'bg-primary/10', text: 'text-primary' };
}

function showErrorFindings(msg) {
    const container = document.getElementById('keyFindings');
    if (!container) return;
    container.innerHTML = `
        <div class="flex items-start gap-3 p-3 bg-accent-danger/10 rounded-xl border border-accent-danger/20">
            <span class="material-symbols-outlined !text-[20px] text-accent-danger mt-0.5">error</span>
            <span class="text-white/80 text-sm">${escapeHtml(msg)}</span>
        </div>`;
}

/* ===================================================================
   AI Explanation panel
   =================================================================== */

function renderExplanation(r) {
    const textEl = document.getElementById('explanationText');
    if (!textEl) return;

    const isFake = r.prediction === 'ai_generated';
    const pct = Math.round(r.fake_probability ?? 50);
    const lines = [];

    if (isFake) {
        lines.push(`The audio was classified as AI-generated with ${pct}% probability.`);
        lines.push(`The ${r.ensemble_details?.[0]?.name || 'detection model'} identified patterns characteristic of synthetic speech generation.`);
    } else {
        lines.push(`The audio was classified as authentic human speech with ${Math.round(r.real_probability ?? 50)}% probability.`);
        lines.push(`The ${r.ensemble_details?.[0]?.name || 'detection model'} confirmed patterns consistent with natural human speech production.`);
    }

    if (r.analysis_mode === 'segmented') {
        const aiSegs = (r.segments || []).filter(s => s.verdict === 'likely_ai').length;
        const total = r.segments_analyzed || r.segments?.length || 0;
        lines.push(`\nSegmented analysis processed ${total} time windows across ${formatDuration(r.total_duration_seconds)} of audio.`);
        if (aiSegs > 0) {
            lines.push(`${aiSegs} of ${total} segments showed AI-generation characteristics.`);
        } else {
            lines.push(`All ${total} segments were consistent with authentic speech.`);
        }
    }

    const artifacts = r.artifacts_detected || [];
    if (artifacts.length) {
        lines.push('\nDetailed findings:');
        artifacts.forEach(a => lines.push(`  • ${a}`));
    }

    textEl.textContent = lines.join('\n');
}

/* ===================================================================
   Detection Model panel
   =================================================================== */

function renderModelDetails(r) {
    const list = document.getElementById('modelsList');
    const subtitle = document.getElementById('modelsSubtitle');
    if (!list) return;

    const model = r.ensemble_details?.[0];
    if (!model) { list.innerHTML = '<p class="text-white/40 text-sm">No model information available.</p>'; return; }

    if (subtitle) subtitle.textContent = model.name;

    list.innerHTML = `
        <div class="bg-white/5 rounded-xl p-4 border border-white/5">
            <div class="flex justify-between items-start mb-3">
                <div>
                    <h5 class="text-white font-bold text-sm">${escapeHtml(model.name)}</h5>
                    <p class="text-white/40 text-xs mt-0.5">${escapeHtml(model.model_id || '')}</p>
                </div>
                <span class="text-[10px] uppercase font-bold px-2 py-1 rounded bg-primary/20 text-primary">Active</span>
            </div>
            <div class="grid grid-cols-2 md:grid-cols-4 gap-3 text-center">
                <div>
                    <p class="text-[10px] text-white/40 uppercase">Fake Prob.</p>
                    <p class="text-sm font-bold ${r.prediction === 'ai_generated' ? 'text-accent-danger' : 'text-accent-success'}">${Math.round(model.fake_probability)}%</p>
                </div>
                <div>
                    <p class="text-[10px] text-white/40 uppercase">Mode</p>
                    <p class="text-sm font-bold text-white">${r.analysis_mode === 'segmented' ? 'Segmented' : 'Single'}</p>
                </div>
                <div>
                    <p class="text-[10px] text-white/40 uppercase">Duration</p>
                    <p class="text-sm font-bold text-white">${formatDuration(r.total_duration_seconds)}</p>
                </div>
                <div>
                    <p class="text-[10px] text-white/40 uppercase">Sample Rate</p>
                    <p class="text-sm font-bold text-white">${r.meta?.sample_rate ? r.meta.sample_rate + ' Hz' : '16000 Hz'}</p>
                </div>
            </div>
        </div>
    `;
}

/* ===================================================================
   Collapsible panels
   =================================================================== */

function setupCollapsiblePanels() {
    setupToggle('explanationToggle', 'explanationContent', 'explanationChevron');
    setupToggle('modelsToggle', 'modelsContent', 'modelsChevron');
}

function setupToggle(btnId, contentId, chevronId) {
    const btn = document.getElementById(btnId);
    const content = document.getElementById(contentId);
    const chevron = document.getElementById(chevronId);
    if (!btn || !content) return;

    btn.addEventListener('click', () => {
        const isHidden = content.classList.contains('hidden');
        content.classList.toggle('hidden');
        if (chevron) chevron.classList.toggle('rotated', isHidden);
    });
}

/* ===================================================================
   Action buttons
   =================================================================== */

function setupActionButtons(result) {
    // Re-analyze
    const reBtn = document.getElementById('reanalyzeBtn');
    if (reBtn) {
        reBtn.addEventListener('click', () => window.location.reload());
    }

    // Export PDF (simple text report)
    const exportBtn = document.getElementById('exportPdfBtn');
    if (exportBtn) {
        exportBtn.addEventListener('click', () => exportReport(result));
    }
}

function exportReport(r) {
    const lines = [
        '═══════════════════════════════════════',
        '  VisioNova — Audio Analysis Report',
        '═══════════════════════════════════════',
        '',
        `Date:       ${new Date().toLocaleString()}`,
        `Verdict:    ${r.prediction === 'ai_generated' ? 'AI-GENERATED' : 'AUTHENTIC'}`,
        `Confidence: ${Math.round(r.confidence)}%`,
        `Fake Prob:  ${Math.round(r.fake_probability)}%`,
        `Real Prob:  ${Math.round(r.real_probability)}%`,
        `Duration:   ${formatDuration(r.total_duration_seconds)}`,
        `Mode:       ${r.analysis_mode}`,
        '',
    ];

    if (r.segments && r.segments.length > 1) {
        lines.push('─── Segment Breakdown ───');
        r.segments.forEach((seg, i) => {
            lines.push(`  #${i + 1}  ${formatDuration(seg.start_sec)}–${formatDuration(seg.end_sec)}  Fake: ${Math.round(seg.fake_probability)}%  ${seg.verdict}`);
        });
        lines.push('');
    }

    if (r.artifacts_detected?.length) {
        lines.push('─── Findings ───');
        r.artifacts_detected.forEach(a => lines.push(`  • ${a}`));
        lines.push('');
    }

    lines.push('─── Model ───');
    lines.push(`  ${r.ensemble_details?.[0]?.name || 'Unknown'}`);
    lines.push(`  ${r.ensemble_details?.[0]?.model_id || ''}`);
    lines.push('');
    lines.push('Generated by VisioNova');

    const blob = new Blob([lines.join('\n')], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `VisioNova_Audio_Report_${Date.now()}.txt`;
    a.click();
    URL.revokeObjectURL(url);
}

/* ===================================================================
   Helpers
   =================================================================== */

function formatDuration(seconds) {
    if (seconds == null || isNaN(seconds)) return '--:--';
    const s = Math.round(seconds);
    const m = Math.floor(s / 60);
    const sec = s % 60;
    if (m >= 60) {
        const h = Math.floor(m / 60);
        const mins = m % 60;
        return `${h}:${String(mins).padStart(2, '0')}:${String(sec).padStart(2, '0')}`;
    }
    return `${m}:${String(sec).padStart(2, '0')}`;
}

function setText(id, text) {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
}

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}
