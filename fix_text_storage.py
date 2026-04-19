
with open('frontend/html/AnalysisDashboard.html', 'r', encoding='utf-8') as f: content = f.read()
import re
for match in re.findall(r'sessionStorage\.setItem\(.*?\)', content): print(match)
