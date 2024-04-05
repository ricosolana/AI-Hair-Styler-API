import argparse
import os
import secrets
import subprocess
import uuid
import json


import cv2
import numpy as np
from flask import Flask, jsonify, request, abort, send_from_directory
from flask_jwt_extended import JWTManager, create_access_token, jwt_required
from werkzeug.utils import secure_filename

# openssl req -x509 -newkey rsa:4096 -nodes -out cert.pem -keyout key.pem -days 365

app = Flask(__name__)

if not app.config.from_file('config.json', load=json.load):
    print('Failed to load config')
    exit(1)

os.environ.setdefault('aihairstyler_JWT_SECRET_KEY', secrets.token_urlsafe(32))
app.config['JWT_SECRET_KEY'] = os.environ['aihairstyler_JWT_SECRET_KEY']

jwt = JWTManager(app)


class WorkStatus:
    def __init__(self, work_id: str, process):
        self.work_id = work_id
        self.process = process


work_queue = {}


# This endpoint is accessible only from localhost
@app.route('/token', methods=['GET'])
def generate_token():
    if request.remote_addr != '127.0.0.1':
        abort(403)  # Forbidden

    # Generate a token without requiring user_id or any other data
    access_token = create_access_token(identity="anonymous")
    print(f'Generated token: {access_token}')
    return jsonify(access_token=access_token)


# this api is protected
@app.route('/api/barber', methods=['POST'])
#@jwt_required()
def api_barber():
    # Check if there is a video stream in the request
    f = request.files.get('image')
    if f is None:
        return jsonify({'message': 'No \'image\' file found in request files'}), 400

    style = request.args.get('style')
    if style is None:
        return jsonify({'message': 'Parameter \'style\' is missing'}), 400

    color = request.args.get('color')
    if color is None:
        return jsonify({'message': 'Parameter \'color\' is missing'}), 400

    style_file_name = app.config['styles'].get(style, None)
    if style_file_name is None:
        return jsonify({'message': f'Hairstyle \'{style}\' is invalid'}), 400

    color_file_name = app.config['colors'].get(color, None)
    if color_file_name is None:
        return jsonify({'message': f'Hair color \'{color}\' is invalid'}), 400

    style_file_name = app.config['input_template_pattern'].format(style_file_name)
    color_file_name = app.config['input_template_pattern'].format(color_file_name)

    work_id = secrets.token_urlsafe(16)

    ext = os.path.splitext(f.filename)[-1].lower()
    if ext not in ('.png', '.jpg', '.jpeg'):
        return jsonify({'message': f'Image extension \'{ext}\' is invalid'})

    input_file_name = work_id + ext
    f.save(os.path.join(app.config['input_directory'], input_file_name))

    process = subprocess.Popen([
        "python", "barber.py",
        '--input_dir', app.config['input_directory'],
        "--im_path1", input_file_name,  # face
        "--im_path2", style_file_name + app.config[''],  # style
        "--im_path3", color_file_name,  # color
        "--sign", "realistic",
        "--smooth", "5",
        "--output_dir", os.path.join(app.config['output_directory'], work_id)
    ])

    work_queue[work_id] = WorkStatus(work_id, process)

    output_file_name = (f'{os.path.splitext(os.path.basename(input_file_name))[0]}_'
                        f'{style}_{color}_realistic.png')

    return jsonify({
        'message': 'Image is now processing',
        'work_id': work_id,
        'name': output_file_name
    })


@app.route('/generated/<path:path>')
@jwt_required()
def serve_outputs(path):
    work_id = request.args.get('work_id')
    if work_id is None:
        return jsonify({'message': 'Parameter \'work_id\' is missing'})

    # prevent directory traversal
    work_id = secure_filename(work_id)
    root = os.path.join(app.config['output_directory'], work_id)

    return send_from_directory(root, path)


"""
#/api/poll
@app.route('/api/barber/poll', methods=['GET'])
@jwt_required()
def api_barber_poll():
    work_id = request.args.get('work_id', None)
    if work_id is None:
        return jsonify({'message': 'Parameter \'work_id\' is missing'}), 400

    # TODO use static directory serving to get image
    #   require that images are authorized / limited to the specified user?



    # save the input image first to disk
    f.save(os.path.join(tmp_input_dir, f.name))

    process = subprocess.Popen([
        "python", "barber.py",
        "--im_path1", "103.png",  # face
        "--im_path2", "28.png",  # style
        "--im_path3", "54.png",  # color
        "--sign", "realistic",
        "--smooth", "5",
        "--output_dir", "my_dir_out"
    ])

    # process.kill() # incase the user wants to signal an cancel?

    original_image = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    conv_image = svoji.process_svoji(original_image, 'svg')

    # TODO remove
    #ress = cv2.imwrite('my_file_from_client.jpeg', original_image)

    if conv_image is None:
        return jsonify({'message': 'Image recognition failed'}), 422

    return jsonify({'image': bytearray(original_image).hex(), 'converted': bytearray(conv_image).hex()})
"""

app.run(host='127.0.0.1', port=80)
#app.run(host='127.0.0.1', port=443, ssl_context=('cert.pem', 'key.pem'))
#app.run(host='192.168.137.1', port=80)
