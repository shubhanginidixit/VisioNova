import re

file_path = 'frontend/html/AnalysisDashboard.html'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Set intervals to 10ms to make it fast
content = re.sub(r'\}, \d+\);', '}, 10);', content)
content = re.sub(r'await new Promise\(r => setTimeout\(r, \d+\)\);', 'await new Promise(r => setTimeout(r, 10));', content)

# 2. Fix the Image QuotaExceededError by deleting big base64 images
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
content = content.replace("sessionStorage.setItem('visioNova_image_result', JSON.stringify(result));", replace_str.strip())

# 3. Change localhost to 127.0.0.1
content = content.replace('http://localhost:5000', 'http://127.0.0.1:5000')

# 4. Remove all dots '...'
content = content.replace('...', '')

# 5. M and dots? Maybe he meant the letters 'M' or 'm' as well?
# "remove all M and dots" -> could be "minutes" or "min"? Or wait, maybe M stands for MegaBytes?
# What if I just remove "M" and "m" from the messages that are shown to the user?
# Actually, the user might mean the capital M and dots from the app interface.
# Let's save this file and ask the user what M means, or just replace 'M' with '' and '.' with '' in certain titles?
# Wait, replacing all 'M' and 'm' will break HTML and JS tags! Please DO NOT replace all 'm's!
# "remove all M and dots text wala after that if u want change alll the M and dots within a sec please"
# Oh... "fix text wala after that if u want change alll the M and dots within a sec please" -> "change all the loading things to happen within a sec" 
# Oh! "change all the Ms and dots to within a sec please" (minutes and dots (...) loading screen)
# "M" = minutes (like 5 Mins, 2 Mins). The fake loading screen takes arbitrary minutes.

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)
