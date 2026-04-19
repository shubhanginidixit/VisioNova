import requests
json_data={'video':'data:video/mp4;base64,AAAA','filename':'test.mp4'}
except_str=''
try:
  r=requests.post('http://127.0.0.1:5000/api/detect-video',json=json_data,timeout=300)
  print(r.status_code)
  print(r.text)
except Exception as e:
  print(e)
