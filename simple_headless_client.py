import requests

url = 'http://localhost:80/api/pose'

#file_path = 'Forward_face.jpg'
file_path = 'test_partial_side.jpg'

with open(file_path, 'rb') as f:
    files = {'frame': f}

    response = requests.post(url, files=files)

print(response.json())
