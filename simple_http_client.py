import requests
import os


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
    #os.environ["REQUESTS_CA_BUNDLE"] = 'C:\\Users\\rico\\Documents\\GitHub\\AI-Hair-Styler-API\\certs\\ca-cert.pem'
    #os.environ["SSL_CERT_FILE"] = 'C:\\Users\\rico\\Documents\\GitHub\\AI-Hair-Styler-API\\certs\\ca-cert.pem'

    access_token = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJmcmVzaCI6ZmFsc2UsImlhdCI6MTcxMjkwMTE2OSwianRpIjoiYWQ3N2I5ZDQtYzA0Mi00Y2NiLWEyMTYtZjQyNDkyZGY3OWVlIiwidHlwZSI6ImFjY2VzcyIsInN1YiI6ImFub255bW91cyIsIm5iZiI6MTcxMjkwMTE2OSwiY3NyZiI6IjMwYThiNGExLTQ2ZmQtNGE5MS1iNTRiLWQzZGM0NzE4N2NkOCIsImV4cCI6MTcxMjkwMjA2OX0.ejyVHwBGxhqeI1g28so64kekVt0mb3WIFaczzYB6Pk8'
    #url = 'http://localhost/api/barber'
    url = 'http://localhost/api/barber'

    headers = {
        'Authorization': f'Bearer {access_token}'
    }

    query_params = {
        'style-file': '1.png',
        'color-file': '2.png'
    }

    files = {'image': img_file}

    response = requests.post(url,
                             headers=headers,
                             files=files,
                             params=query_params,
                             #verify='C:\\Users\\rico\\Downloads\\O=periodicseizures,L=Tampa,ST=Florida,C=US.crt'
                             #verify='C:\\Users\\rico\\Downloads\\aihairstyler.crt'
                             #verify=False
                             )
    if response.status_code == 200:
        j = response.json()
        print(j['work-id'])
    elif response.status_code == 422:
        print('invalid access token, retrieve another!')
    else:
        print(response.json())


if __name__ == '__main__':
    # test_api_pose()

    #test_api_svoji()
    #file_path = 'rect-image.jpeg'
    file_path = 'square-image.jpg'

    with open(file_path, 'rb') as img_file:
        #for i in range(20):
        test_api_barber(img_file)
