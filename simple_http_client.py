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


def test_api_barber():
    access_token = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJmcmVzaCI6ZmFsc2UsImlhdCI6MTcxMjM3Mzk0OCwianRpIjoiYTdmZTY4NDUtNmU5OC00ZDA5LTlmNjctNTBmYThmYWIzNmU4IiwidHlwZSI6ImFjY2VzcyIsInN1YiI6ImFub255bW91cyIsIm5iZiI6MTcxMjM3Mzk0OCwiY3NyZiI6ImJmZjU2MTI5LTczZDMtNDM3Ni1hNmRhLTA2NjRmYzJkZGU2NSIsImV4cCI6MTcxMjM3NDg0OH0.kI2kY0YyJKkSuoIgCkXS5zmJgahe2gsf1d-bVKnWXwk'
    url = 'http://localhost:80/api/barber'

    headers = {
        'Authorization': f'Bearer {access_token}'
    }

    # file_path = 'Forward_face.jpg'
    file_path = 'test_partial_side.jpg'

    query_params = {
        'style': 'bob',
        'color': 'dark-blonde'
    }

    with open(file_path, 'rb') as f:
        files = {'image': f}

        response = requests.post(url,
                                 headers=headers,
                                 files=files,
                                 params=query_params)

    print(response.json())


if __name__ == '__main__':
    # test_api_pose()

    #test_api_svoji()

    test_api_barber()
