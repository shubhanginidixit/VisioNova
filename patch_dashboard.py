import re
import sys

def main():
    file_path = 'frontend/html/AnalysisDashboard.html'
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print("Error reading:", e)
        return

    # Speed up intervals and timeouts to be instantaneous ("within a sec")
    # setInterval(..., 300) -> setInterval(..., 10)
    content = re.sub(r'setInterval\(\(\) => \{([\s\S]*?)if \(progressValue < \d+\) \{[\s\S]*?progressValue \+= \d+;[\s\S]*?updateStepProgress\([\s\S]*?\);[\s\S]*?\}[\s\S]*?\}, \d+\);', 
                     lambda m: m.group(0).replace(re.search(r'\}, (\d+)\);', m.group(0)).group(1), '10').replace(re.search(r'progressValue \+= (\d+);', m.group(0)).group(1), '20'), 
                     content)

    # All simple setTimeouts that are just delays
    content = re.sub(r'await new Promise\(r => setTimeout\(r, \d+\)\);', 'await new Promise(r => setTimeout(r, 10));', content)

    # Also fix the actual setInterval calls directly
    content = re.sub(r'const progressInterval = setInterval\(\(\) => \{\s*if \(progressValue < \d+\) \{\s*progressValue \+= \d+;\s*updateStepProgress\(.*?\);\s*\}\s*\}, \d+\);',
        lambda m: m.group(0).replace(re.findall(r'\}, (\d+)\);', m.group(0))[-1], '10').replace(re.findall(r'progressValue \+= (\d+)', m.group(0))[-1], '50'),
        content)

    content = re.sub(r'setTimeout\(\(\) => \{\s*updateStep\(\);\s*\}, \d+\);', 'setTimeout(() => { updateStep(); }, 10);', content)
    content = re.sub(r'setTimeout\(updateStep, \d+\);', 'setTimeout(updateStep, 10);', content)

    # Change "mins" and "min" in the UI text to not show minutes
    content = content.replace("5 mins for large docs", "instant")
    content = content.replace("5 mins for initial model download", "instant")
    content = content.replace("3 min for audio", "instant")
    content = content.replace("3 min for video", "instant")

    # The user wanted "M and dots" removed (Processing..., min, etc.)
    # We can replace '...' with '' in the status messages
    content = content.replace("Processing...", "Processing")
    content = content.replace("Extracting metadata...", "Extracting metadata")
    content = content.replace("Detecting AI artifacts...", "Detecting AI artifacts")
    content = content.replace("Analyzing pixel patterns...", "Analyzing pixel patterns")
    content = content.replace("Analyzing audio...", "Analyzing audio")
    content = content.replace("Analyzing frames...", "Analyzing frames")
    content = content.replace("Processing document...", "Processing document")
    content = content.replace("Extracting content...", "Extracting content")
    content = content.replace("Verifying claims...", "Verifying claims")
    content = content.replace("Tokenizing text content...", "Tokenizing text content")
    content = content.replace("Analyzing linguistic patterns...", "Analyzing linguistic patterns")
    content = content.replace("Checking source attribution...", "Checking source attribution")
    content = content.replace("Checking for deepfakes...", "Checking for deepfakes")

    # Fix QuotaExceededError for image storage by stripping base64 images before sessionStorage
    replace_str = """
                                    if (result.individual_results && result.individual_results.ela) {
                                        delete result.individual_results.ela.ela_image;
                                    }
                                    if (result.individual_results && result.individual_results.watermark && result.individual_results.watermark.visualizations) {
                                        delete result.individual_results.watermark.visualizations.watermark_heatmap;
                                        delete result.individual_results.watermark.visualizations.fft_magnitude;
                                    }
                                    sessionStorage.setItem('visioNova_image_result', JSON.stringify(result));
"""
    if "sessionStorage.setItem('visioNova_image_result', JSON.stringify(result));" in content:
        content = content.replace("sessionStorage.setItem('visioNova_image_result', JSON.stringify(result));", replace_str.strip())
    
    # Change API_BASE_URL locally
    content = content.replace("'http://localhost:5000'", "'http://127.0.0.1:5000'")

    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("Updated AnalysisDashboard.html successfully.")
    except Exception as e:
        print("Error writing:", e)

if __name__ == '__main__':
    main()
