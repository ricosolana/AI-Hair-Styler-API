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
    url = 'http://localhost:80/api/barber'

    # file_path = 'Forward_face.jpg'
    file_path = 'test_partial_side.jpg'

    query_params = {
        'style': 'bob',
        'color': 'dark-blonde'
    }

    with open(file_path, 'rb') as f:
        files = {'image': f}

        response = requests.post(url, files=files, params=query_params)

    print(response.json())


if __name__ == '__main__':
    # test_api_pose()

    #test_api_svoji()

    test_api_barber()
