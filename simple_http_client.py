import requests


def test_api_pose():
    url = 'http://localhost:80/api/pose'

    file_path = 'test_partial_side.jpg'

    with open(file_path, 'rb') as f:
        files = {'frame': f}

        response = requests.post(url, files=files)

    print(response.json())


def test_api_svoji():
    url = 'http://localhost:80/api/svoji'

    # file_path = 'Forward_face.jpg'
    file_path = 'test_partial_side.jpg'

    with open(file_path, 'rb') as f:
        files = {'image': f}

        response = requests.post(url, files=files)

    print(response.json())


def test_api_barber(img_file):
    access_token = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJmcmVzaCI6ZmFsc2UsImlhdCI6MTcxMjM4NDQ2OCwianRpIjoiMThiYTcxZjAtYTA3Ny00MzRhLWIxNjktMjBkYzhhZDkwMThhIiwidHlwZSI6ImFjY2VzcyIsInN1YiI6ImFub255bW91cyIsIm5iZiI6MTcxMjM4NDQ2OCwiY3NyZiI6Ijg5MzBmMDMwLWQyYzUtNDA4MS04Nzk5LWRkZGIwNjY5NWNkOCIsImV4cCI6MTcxMjM4NTM2OH0.XThOc1amUcMDEiectSh3sZcYNsvrt-TqaZIKtEFHf-I'
    #url = 'https://localhost/api/barber'
    url = 'http://localhost/api/barber'

    headers = {
        'Authorization': f'Bearer {access_token}'
    }

    query_params = {
        'style': 'bob',
        'color': 'dark-blonde'
    }

    files = {'image': img_file}

    response = requests.post(url,
                             headers=headers,
                             files=files,
                             params=query_params)

    print(response.json())


if __name__ == '__main__':
    # test_api_pose()

    #test_api_svoji()
    file_path = 'test_partial_side.jpg'

    with open(file_path, 'rb') as img_file:
        for i in range(20):
            test_api_barber(img_file)
