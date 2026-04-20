

// Kick off once the page is ready
document.addEventListener('DOMContentLoaded', async () => {
    try {
        const videoData = await getVideoFromStorage();
        if (!videoData) {
            showError('No video provided. Please upload a video on the homepage.');
            return;
        }

        showVideoPreview(videoData);

        // Prefer cached analysis from the dashboard
        const cached = await readCachedResult();
        const result = cached || await analyzeVideo(videoData);

        if (result) {
            renderResult(result);
        } else {
            showError('Unable to analyze video.');
        }
    } catch (err) {
        console.error('[VideoResult] Error:', err);
        showError(err.message || 'Unexpected error while processing video.');
    }
});

async function getVideoFromStorage() {
    try {
        if (window.VisioNovaStorage) {
            const dbVideo = await VisioNovaStorage.getVideoFile();
            if (dbVideo && dbVideo.data) return dbVideo;
            
            if (typeof VisioNovaStorage.getFile === 'function') {
                return VisioNovaStorage.getFile('video');
            }
        }
    } catch (e) {
        console.warn('[VideoResult] Storage unavailable:', e);
    }
    return null;
}

function showVideoPreview(videoData) {
    const placeholder = document.getElementById('noVideoPlaceholder');
    const player = document.getElementById('uploadedVideo');

    if (player && videoData.data) {
        player.src = videoData.data;
        player.classList.remove('hidden');
    }
    placeholder?.classList.add('hidden');

    if (videoData.fileName) {
        setText('#pageTitle', `Video Analysis: ${videoData.fileName}`);
    }
}

async function readCachedResult() {
    const cached = await VisioNovaStorage.getResult('video');
    return cached || null;
}
}

async function analyzeVideo(videoData) {
    if (!videoData?.data) {
        throw new Error('Video data missing.');
    }

    const response = await fetch(`${API_BASE_URL}/api/detect-video`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            video: videoData.data,
            filename: videoData.fileName || 'video.mp4'
        })
    });

    if (!response.ok) {
        const errText = await response.text().catch(() => '');
        throw new Error(errText || `API error ${response.status}`);
    }

    return await response.json();
}

function renderResult(result) {
    const aiProb = pickNumber(result.ai_probability, result.fake_probability, result.confidence, 0);
    const humanProb = pickNumber(result.human_probability, 100 - aiProb);
    const isFake = (result.prediction || '').toLowerCase().includes('fake') || aiProb >= humanProb;

    updateScoreCard(aiProb, isFake);
    updateSummaryText(result, isFake);
    updateMetadata(result);
    updateBadges(isFake, result);
}

function updateScoreCard(aiProb, isFake) {
    const scoreValueEl = document.querySelector('.size-48 span.text-5xl, .size-48 span.text-4xl');
    if (scoreValueEl) scoreValueEl.textContent = `${Math.round(aiProb)}%`;

    const labelEl = document.querySelector('.size-48 span.text-xs');
    if (labelEl) {
        labelEl.textContent = isFake ? 'Fake Probability' : 'Authenticity';
        labelEl.classList.toggle('text-visio-accent-red', isFake);
        labelEl.classList.toggle('text-accent-success', !isFake);
    }

    const circle = document.querySelector('svg circle.text-visio-accent-red, svg circle.text-visio-accent-blue, svg circle.text-visio-accent-amber');
    if (circle) {
        const r = Number(circle.getAttribute('r')) || 42;
        const circumference = 2 * Math.PI * r;
        const offset = circumference - (Math.min(Math.max(aiProb, 0), 100) / 100) * circumference;
        circle.setAttribute('stroke-dashoffset', offset.toFixed(2));
        circle.setAttribute('stroke', isFake ? '#FF4A4A' : '#00D991');
    }
}

function updateSummaryText(result, isFake) {
    const summaryEl = document.querySelector('.w-full.bg-[#0f121a].rounded-lg p.text-sm');
    if (summaryEl) {
        const verdict = isFake ? 'AI-generated content detected' : 'Content appears authentic';
        const model = result.model ? `Detector: ${result.model}` : '';
        const fakeFrames = result.fake_frame_count ? `${result.fake_frame_count} of ${result.frame_count || 'n/a'} frames flagged` : '';
        summaryEl.textContent = [verdict, model, fakeFrames].filter(Boolean).join(' • ');
    }
}

function updateMetadata(result) {
    const statRows = document.querySelectorAll('.flex.justify-between.items-center.text-sm');
    if (statRows.length >= 3) {
        // Source row
        const sourceVal = statRows[0].querySelector('.text-white');
        if (sourceVal) sourceVal.textContent = result.model || result.verdict || 'Video source';

        // Resolution row
        const resVal = statRows[1].querySelector('.text-white');
        if (resVal) {
            const res = result.resolution || result.video_info?.resolution;
            resVal.textContent = res || 'Resolution unknown';
        }

        // Audio codec row
        const codecVal = statRows[2].querySelector('.text-white');
        if (codecVal) {
            const codec = result.video_info?.audio_codec || result.video_info?.codec;
            codecVal.textContent = codec || 'Codec unknown';
        }
    }
}

function updateBadges(isFake, result) {
    // Header badge
    const badge = document.querySelector('.px-2.py-0.5.rounded.text-[10px].font-bold');
    if (badge) {
        badge.textContent = isFake ? 'High Risk Detected' : 'Authenticity Confirmed';
        badge.classList.toggle('bg-visio-accent-red/20', isFake);
    }

    // Timeline header numbers (optional)
    const timelineLabels = document.querySelectorAll('.flex.items-center.justify-between.text-xs.text-gray-400.font-mono span');
    if (timelineLabels.length >= 2 && result.duration_seconds) {
        timelineLabels[0].textContent = '00:00.00';
        timelineLabels[1].textContent = formatDuration(result.duration_seconds);
    }
}

function showError(message) {
    const placeholder = document.getElementById('noVideoPlaceholder');
    if (placeholder) {
        placeholder.classList.remove('hidden');
        placeholder.querySelector('p')?.textContent = message;
    }
}

function setText(selector, text) {
    const el = document.querySelector(selector);
    if (el) el.textContent = text;
}

function pickNumber(...values) {
    for (const v of values) {
        if (v === 0 || v) return Number(v);
    }
    return 0;
}

function formatDuration(seconds) {
    const sec = Math.max(0, Number(seconds) || 0);
    const mins = Math.floor(sec / 60);
    const rem = Math.floor(sec % 60);
    return `${String(mins).padStart(2, '0')}:${String(rem).padStart(2, '0')}`;
}
