import requests
import json

# Test 1: Human text
r1 = requests.post('http://127.0.0.1:5000/api/detect-ai', json={'text': 'hey whats up buddy'})
data1 = r1.json()
print('TEST 1 - Human text:')
print(f'  Prediction: {data1["prediction"]}')
print(f'  AI Score: {data1["scores"]["ai_generated"]:.1f}%')
print()

# Test 2: AI formal text  
r2 = requests.post('http://127.0.0.1:5000/api/detect-ai', json={'text': 'In conclusion, the multifaceted implications of this phenomenon warrant comprehensive examination of underlying mechanisms and their systematic integration within existing frameworks.'})
data2 = r2.json()
print('TEST 2 - AI formal text:')
print(f'  Prediction: {data2["prediction"]}')
print(f'  AI Score: {data2["scores"]["ai_generated"]:.1f}%')
