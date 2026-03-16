/**
 * VisioNova Audio Result Page
 * Handles dynamic audio content loading and real-time ensemble analysis.
 */

document.addEventListener('DOMContentLoaded', function () {
    const audioData = VisioNovaStorage.getFile('audio');

    if (audioData) {
        updateElement('pageTitle', audioData.fileName);

        // Display the uploaded audio
        const placeholder = document.getElementById('noAudioPlaceholder');
        const uploadedAudio = document.getElementById('uploadedAudio');
        const spectrogramBg = document.getElementById('spectrogramBg');

        if (placeholder && uploadedAudio) {
            placeholder.style.display = 'none';
            uploadedAudio.src = audioData.data;
            uploadedAudio.style.display = 'block';

            // Simple visualizer effect (mock for now, replaces static image)
            if (spectrogramBg) {
                spectrogramBg.style.opacity = '0.5';
            }
        }

        // Start Analysis
        runAudioAnalysis(audioData);
    } else {
        // Redirect if no data
        window.location.href = 'AnalysisDashboard.html';
    }
});

async function runAudioAnalysis(audioData) {
    try {
        const response = await fetch('/api/detect-audio', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                audio: audioData.data,
                filename: audioData.fileName
            })
        });

        if (!response.ok) {
            throw new Error(`Analysis failed: ${response.statusText}`);
        }

        const result = await response.json();
        updateUI(result);

    } catch (error) {
        console.error("Audio analysis error:", error);
        updateElement('verdictLabel', 'ERROR');
        updateElement('verdictDescription', 'Failed to analyze audio. Please try again.');
    }
}

function updateUI(result) {
    // 1. Authenticity Score & Verdict
    const isFake = result.prediction === 'ai_generated';
    const confidence = result.confidence; // 0-100
    const aiProb = result.ai_probability || (isFake ? confidence : 100 - confidence);

    // Animate Score
    animateValue("authenticityScore", 0, isFake ? Math.round(100 - aiProb) : Math.round(confidence), 1500);

    // Verdict Label
    const verdictLabel = document.getElementById('verdictLabel');
    const fakeProbLabel = document.getElementById('fakeProbability');
    const progressBar = document.getElementById('scoreProgressBar');
    const verdictDesc = document.getElementById('verdictDescription');

    if (isFake) {
        verdictLabel.textContent = "SUSPICIOUS";
        verdictLabel.className = "text-accent-red font-bold";
        fakeProbLabel.textContent = `${Math.round(aiProb)}% FAKE`;
        fakeProbLabel.className = "text-accent-red font-bold";

        progressBar.style.width = `${aiProb}%`;
        progressBar.className = "h-full bg-gradient-to-r from-accent-red to-orange-500 rounded-full shadow-[0_0_10px_rgba(255,74,74,0.5)] transition-all duration-1000";

        verdictDesc.textContent = "AI-generated artifacts detected by ensemble models.";
    } else {
        verdictLabel.textContent = "AUTHENTIC";
        verdictLabel.className = "text-accent-green font-bold";
        fakeProbLabel.textContent = `${Math.round(result.human_probability || 100 - aiProb)}% REAL`;
        fakeProbLabel.className = "text-accent-green font-bold";

        progressBar.style.width = `${result.human_probability || 100 - aiProb}%`;
        progressBar.className = "h-full bg-gradient-to-r from-accent-green to-emerald-500 rounded-full shadow-[0_0_10px_rgba(74,255,120,0.5)] transition-all duration-1000";

        verdictDesc.textContent = "Audio appears to be genuine human speech.";
    }

    // 2. Ensemble Details
    const ensembleText = document.getElementById('ensembleModelText');
    const ensembleScore = document.getElementById('ensembleAgreementScore');

    if (result.ensemble_details && result.ensemble_details.length > 0) {
        // Construct detailed string
        const details = result.ensemble_details.map(m => `${m.name}: ${m.fake_probability}% Fake`).join(' | ');
        ensembleText.textContent = details;
        ensembleText.title = details; // Tooltip

        // Calculate agreement (std dev or simply range)
        const probs = result.ensemble_details.map(m => m.fake_probability);
        const range = Math.max(...probs) - Math.min(...probs);
        const agreement = range < 20 ? "High" : (range < 50 ? "Medium" : "Low");
        ensembleScore.textContent = `${agreement} Agreement`;
    } else {
        ensembleText.textContent = "Single Model Analysis";
        ensembleScore.textContent = "N/A";
    }

    // 3. Artifacts / Deep Analysis
    // Mapping backend artifacts to UI elements
    const artifacts = result.artifacts_detected || [];

    // Pitch (Frequency consistency)
    const pitchElem = document.getElementById('pitchStability');
    if (pitchElem) {
        if (artifacts.some(a => a.toLowerCase().includes('frequency') || a.toLowerCase().includes('spectral'))) {
            pitchElem.textContent = "Unstable";
            pitchElem.className = "text-2xl font-bold text-accent-amber mb-1";
        } else {
            pitchElem.textContent = "Natural";
            pitchElem.className = "text-2xl font-bold text-accent-green mb-1";
        }
    }

    // Background Noise
    const noiseElem = document.getElementById('bgNoiseLevel');
    if (noiseElem) {
        // Check for "silence" or digital zero artifacts
        if (artifacts.some(a => a.toLowerCase().includes('vocoder') || a.toLowerCase().includes('silence'))) {
            noiseElem.textContent = "Artificial";
            noiseElem.className = "text-2xl font-bold text-accent-red mb-1";
        } else {
            noiseElem.textContent = "Natural Floor";
            noiseElem.className = "text-2xl font-bold text-accent-green mb-1";
        }
    }

    // Formant Analysis
    const formantElem = document.getElementById('formantAnalysis');
    if (formantElem) {
        if (isFake) {
            formantElem.textContent = "Synthetic";
            formantElem.className = "text-2xl font-bold text-accent-red mb-1";
        } else {
            formantElem.textContent = "Consistent";
            formantElem.className = "text-2xl font-bold text-accent-green mb-1";
        }
    }
}

function updateElement(id, text) {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
}

function animateValue(id, start, end, duration) {
    const obj = document.getElementById(id);
    if (!obj) return;

    let startTimestamp = null;
    const step = (timestamp) => {
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        obj.textContent = Math.floor(progress * (end - start) + start);
        if (progress < 1) {
            window.requestAnimationFrame(step);
        }
    };
    window.requestAnimationFrame(step);
}
