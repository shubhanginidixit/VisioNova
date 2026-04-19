

print("Applying instant fixes...")

# 1. Ensure all URLs point to 127.0.0.1 to avoid CORS / connection refusal issues
files = glob.glob('frontend/**/*.html', recursive=True) + glob.glob('frontend/**/*.js', recursive=True)
for f in files:
    try:
        with open(f, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Replace localhost to 127.0.0.1
        new_content = content.replace('http://localhost:5000', 'http://127.0.0.1:5000')
        new_content = new_content.replace('http://localhost:5000/', 'http://127.0.0.1:5000/')
        
        if content != new_content:
            with open(f, 'w', encoding='utf-8') as file:
                file.write(new_content)
    except Exception as e:
        print(f"Skipping {f}: {e}")

# 2. Fix Dashboard (Speed + remove '...', remove 'min'/'Mins', ensure no QuotaException)
dash = 'frontend/html/AnalysisDashboard.html'
try:
    with open(dash, 'r', encoding='utf-8') as f:
        c = f.read()

    # Make visual pauses extremely fast
    c = re.sub(r'await new Promise\(\s*r\s*=>\s*setTimeout\(\s*r\s*,\s*\d+\s*\)\s*\);', 'await new Promise(r => setTimeout(r, 10));', c)
    c = re.sub(r'progressValue \+= \d+;', 'progressValue += 50;', c)
    
    # Modify the interval speed but safely target only the progress intervals
    c = re.sub(r'setInterval\(\(\) => \{([\s\S]*?if \(progressValue < \d+\) \{[\s\S]*?updateStepProgress.*?\})[\s\S]*?\}, \d+\);',
               r'setInterval(() => {\1}, 10);', c)

    # Remove dots and Mins
    c = c.replace('...', '')
    c = re.sub(r'\b\d+\s*mins?\b', 'a sec', c, flags=re.IGNORECASE)
    
    # Strip bloat from base64 heatmaps that causes QuotaExceededError in sessionStorage
    if 'delete result.individual_results.ela.ela_image' not in c:
        safe_save = """
                                    if (result.individual_results && result.individual_results.ela) {
                                        delete result.individual_results.ela.ela_image;
                                    }
                                    if (result.individual_results && result.individual_results.watermark && result.individual_results.watermark.visualizations) {
                                        delete result.individual_results.watermark.visualizations.watermark_heatmap;
                                        delete result.individual_results.watermark.visualizations.fft_magnitude;
                                    }
                                    sessionStorage.setItem('visioNova_image_result', JSON.stringify(result));"""
        c = c.replace("sessionStorage.setItem('visioNova_image_result', JSON.stringify(result));", safe_save.strip())

    with open(dash, 'w', encoding='utf-8') as f:
        f.write(c)

except Exception as e:
    print(f"Error modifying Dashboard: {e}")

print("Fixes applied.")
