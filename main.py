import json

import cv2
import numpy as np
from flask import Flask, jsonify, request, Response
from flask_socketio import SocketIO
from requests_toolbelt import MultipartEncoder
import binascii

import proc
import svoji

#sio = socketio.Server(async_mode='threading')
app = Flask(__name__)
#app.config['JWT_SECRET_KEY'] = os.environ.get('ai_jwt_key', 'hamburgers-from-krystal-are-very-good-and-delicious-and-fluffy')
app.config['SECRET_KEY'] = 'secret'
#app.wsgi_app = socketio.WSGIApp(sio, app.wsgi_app)
socketio = SocketIO(app, logger=True, engineio_logger=True)
#####app.config['JWT_SECRET_KEY'] = 'hamburgers-from-krystal-are-very-good-and-delicious-and-fluffy'  # Change this!
#jwt = JWTManager(app)

"""
# This endpoint is accessible only from localhost
@app.route('/generate-token', methods=['GET'])
def generate_token():
    if request.remote_addr != '127.0.0.1':
        abort(403)  # Forbidden

    # Generate a token without requiring user_id or any other data
    access_token = create_access_token(identity="anonymous")
    print(f'Generated token: {access_token}')
    return jsonify(access_token=access_token)
"""

"""
@socketio.on('/api/pose_feed')
def api_pose_feed():


    return Response()
"""


@socketio.on('/api/svoji')
def event_api_svoji(sid, data):
    original_image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    conv_image = svoji.process_svoji(original_image, 'svg')
    return "OK", bytearray(conv_image)


@socketio.event
#def connect(sid, environ, auth):
def connect(sid):
    # authenticate the client
    return True



@app.route('/api/svoji', methods=['POST'])
def api_svoji():
    # Check if there is a video stream in the request
    f = request.files.get('image')
    if f is None:
        return jsonify({'message': 'No \'image\' found in request'}), 400

    original_image = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    conv_image = svoji.process_svoji(original_image, 'svg')

    # TODO remove
    #ress = cv2.imwrite('my_file_from_client.jpeg', original_image)

    if conv_image is None:
        return jsonify({'message': 'Image recognition failed'}), 422

    return jsonify({'image': bytearray(original_image).hex(), 'converted': bytearray(conv_image).hex()})



@app.route('/api/pose', methods=['POST'])
def api_pose():
    while True:
        # Check if there is a video stream in the request
        if 'frame' not in request.files:
            return jsonify({'message': 'No frame found in request'}), 400

        f = request.files['frame']

        if f:
            #image = cv2.imdecode(np.fromstring(f.read(), np.uint8), cv2.IMREAD_UNCHANGED)
            image = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_UNCHANGED)
            (json_data, img) = proc.process_frame_pose(image)

            if img is None:
                return jsonify(json_data)

            m = MultipartEncoder(
                fields={
                    'image': ('cropped', img, 'image/jpeg'),
                    #'data': json.dumps(json_data)
                    'data': ('data.json', json.dumps(json_data), 'application/json')
                }
            )

            return Response(m.to_string(), mimetype=m.content_type)

            """
            # multipart
            def generate():
                yield b'--boundary\r\nContent-Disposition: form-data; name="image"\r\nContent-Type: image/jpeg\r\n\r\n'
                yield img
                yield b'\r\n--boundary\r\nContent-Disposition: form-data; name="data"\r\nContent-Type: application/json\r\n\r\n'
                yield json.dumps(json_data).encode('utf-8')
                yield b'\r\n--boundary--'

            # Set the response headers
            headers = {
                'Content-Type': 'multipart/form-data; boundary=boundary'
            }

            return Response(generate(), headers=headers)
            """



            #return jsonify(data)

            #height = image.shape[0]
            #width = image.shape[1]
            #return jsonify({'height': height, 'width': width}), 200


if __name__ == '__main__':
    socketio.run(app, host='127.0.0.1', port=80, allow_unsafe_werkzeug=True)
    #app.run(host='127.0.0.1', port=80)
    #app.run(host='192.168.137.1', port=80)
