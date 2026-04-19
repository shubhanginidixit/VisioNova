import re
import sys

def main():
    file_path = 'frontend/html/AnalysisDashboard.html'
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Speed up all setIntervals
    content = re.sub(r'\}, \d+\);', '}, 10);', content)
    
    # Strip bloat from image result before saving
    strip_payload = """
                                    // Strip large unused heatmaps to prevent QuotaExceededError
                                    if (result.individual_results && result.individual_results.ela) {
                                        delete result.individual_results.ela.ela_image;
                                    }
                                    if (result.individual_results && result.individual_results.watermark && result.individual_results.watermark.visualizations) {
                                        delete result.individual_results.watermark.visualizations.watermark_heatmap;
                                        delete result.individual_results.watermark.visualizations.fft_magnitude;
                                    }

                                    // Store result for ImageResultPage to use
                                    sessionStorage.setItem('visioNova_image_result', JSON.stringify(result));
"""
    content = content.replace("sessionStorage.setItem('visioNova_image_result', JSON.stringify(result));", strip_payload.strip())

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == "__main__":
    main()