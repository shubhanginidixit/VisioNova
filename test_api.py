import requests
print(requests.post('http://localhost:5000/api/detect-image/ensemble', json={'image':'dummy'}).json())
