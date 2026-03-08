/**
 * VisioNova Video Result Page
 * Handles dynamic video content loading and mock analysis results
 */

document.addEventListener('DOMContentLoaded', function () {
    const videoData = VisioNovaStorage.getFile('video');

    if (videoData) {
        updateElement('pageTitle', 'Video Analysis: ' + videoData.fileName);

        // Display the uploaded video
        const placeholder = document.getElementById('noVideoPlaceholder');
        const uploadedVideo = document.getElementById('uploadedVideo');

        if (uploadedVideo) {
            uploadedVideo.src = videoData.data;
            uploadedVideo.classList.remove('hidden');
        }
        if (placeholder) {
            placeholder.classList.add('hidden');
        }

        // Generate and display mock analysis results
        displayVideoAnalysis(videoData);
    } else {
        updateElement('pageTitle', 'Video Analysis');
    }
});

function displayVideoAnalysis(fileData) {
    const hash = hashString(fileData.fileName);
    const authenticityScore = 35 + (hash % 60);
    const fakeScore = 100 - authenticityScore;
    const isFake = fakeScore > 50;

    // Update overall score
    const scoreElement = document.querySelector('.text-5xl.font-black, .text-4xl.font-black');
    if (scoreElement) {
        scoreElement.textContent = authenticityScore;
    }

    // Update score circle color
    const scoreCircle = document.querySelector('circle[stroke="#00D991"], circle[stroke="#FF4A4A"]');
    if (scoreCircle) {
        const circumference = 251.2;
        const offset = circumference - (authenticityScore / 100) * circumference;
        scoreCircle.setAttribute('stroke-dashoffset', offset);
        scoreCircle.setAttribute('stroke', isFake ? '#FF4A4A' : '#00D991');
    }

    // Update fake probability display
    const fakeProbText = document.querySelector('.text-accent-red.font-black, .text-accent-green.font-black');
    if (fakeProbText) {
        fakeProbText.textContent = fakeScore + '%';
        fakeProbText.className = fakeProbText.className.replace(/text-accent-(red|green)/g,
            isFake ? 'text-accent-red' : 'text-accent-green');
    }

    // Update duration and metadata
    const durationEl = document.querySelector('[class*="Duration"]');
    if (durationEl) {
        const mins = 1 + (hash % 5);
        const secs = hash % 60;
        durationEl.textContent = `${mins}:${secs.toString().padStart(2, '0')}`;
    }

    // Animate analysis bars
    animateProgressBars(hash);

    // Update frame counts
    const totalFrames = 24 * (60 + hash % 120);
    updateElement('totalFrames', totalFrames.toLocaleString());
    updateElement('analyzedFrames', Math.floor(totalFrames * 0.15).toLocaleString());
}

function animateProgressBars(hash) {
    const bars = document.querySelectorAll('.bg-accent-blue, .bg-primary, .bg-green-500, .bg-yellow-500');
    bars.forEach((bar, index) => {
        const width = 25 + ((hash + index * 23) % 70);
        bar.style.transition = 'width 1.2s ease-out';
        setTimeout(() => { bar.style.width = width + '%'; }, index * 200);
    });
}

function updateElement(id, text) {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
}

function hashString(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
        hash = ((hash << 5) - hash) + str.charCodeAt(i);
        hash = hash & hash;
    }
    return Math.abs(hash);
}
