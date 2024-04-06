import json
import os
import secrets
import subprocess
import queue
import threading
import time

from flask import Flask, jsonify, request, abort, send_from_directory
from flask_jwt_extended import JWTManager, create_access_token, jwt_required
from werkzeug.utils import secure_filename

# hmm idk
# openssl req -x509 -newkey rsa:4096 -nodes -out cert.pem -keyout key.pem -days 365


app = Flask(__name__)

if not app.config.from_file('config.json', load=json.load):
    print('Failed to load config')
    exit(1)

BARBER_MAIN = app.config['BARBER_MAIN']
BARBER_INPUT_DIRECTORY = app.config['BARBER_INPUT_DIRECTORY']
SERVING_OUTPUT_DIRECTORY = app.config['SERVING_OUTPUT_DIRECTORY']


app.config["JWT_TOKEN_LOCATION"] = ["headers", "cookies"]
#app.config['JWT_COOKIE_SECURE'] = True  # cookies over https only
app.config['JWT_SECRET_KEY'] = os.environ.setdefault('JWT_SECRET_KEY', secrets.token_urlsafe(32))
jwt = JWTManager(app)

# thread-safe
task_queue = queue.LifoQueue()


def worker():
    while True:
        # blocks until item available, pops it
        task = task_queue.get()
        # I must insert the None
        if task is None:
            break

        subprocess.run(task)
        task_queue.task_done()


worker_thread = threading.Thread(target=worker)
worker_thread.start()


# This endpoint is accessible only from localhost
@app.route('/', methods=['GET'])
def index_path():
    return jsonify({'message': '/api/token, /api/barber, /generated/21_45_24_realistic.png?work_id=8328723823-23912838232'})


# This endpoint is accessible only from localhost
@app.route('/api/token', methods=['GET'])
def api_token():
    if request.remote_addr != '127.0.0.1':
        abort(403)  # Forbidden

    # Generate a token without requiring user_id or any other data
    access_token = create_access_token(identity="anonymous")
    print(f'Generated token: {access_token}')
    return jsonify(access_token=access_token)


def run_barber_process(input_dir, im_path1, im_path2, im_path3, sign, output_dir):
    task_queue.put([
        "python", BARBER_MAIN,
        '--input_dir', input_dir,
        "--im_path1", im_path1,  # face
        "--im_path2", im_path2,  # style
        "--im_path3", im_path3,  # color
        "--sign", sign,
        "--smooth", "5",
        "--output_dir", output_dir  # work_output_directory
    ])

    im_path1 = os.path.join(input_dir, im_path1)
    im_path2 = os.path.join(input_dir, im_path2)
    im_path3 = os.path.join(input_dir, im_path3)

    im_name_1 = os.path.splitext(os.path.basename(im_path1))[0]
    im_name_2 = os.path.splitext(os.path.basename(im_path2))[0]
    im_name_3 = os.path.splitext(os.path.basename(im_path3))[0]

    return '{}_{}_{}_{}.png'.format(im_name_1, im_name_2, im_name_3, sign)


# this api is protected
@app.route('/api/barber', methods=['POST'])
@jwt_required()
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

    style_file_name = app.config['STYLES'].get(style)
    if style_file_name is None:
        return jsonify({'message': f'Hairstyle \'{style}\' is invalid'}), 400

    color_file_name = app.config['COLORS'].get(color)
    if color_file_name is None:
        return jsonify({'message': f'Hair color \'{color}\' is invalid'}), 400

    work_id = secrets.token_hex(32)

    ext = os.path.splitext(f.filename)[-1].lower()
    if ext not in ('.png', '.jpg', '.jpeg'):
        return jsonify({'message': f'Image extension \'{ext}\' is invalid'})

    input_file_name = work_id + ext
    os.makedirs(BARBER_INPUT_DIRECTORY, exist_ok=True)
    f.save(os.path.join(BARBER_INPUT_DIRECTORY, input_file_name))

    # Served image is physically saved to
    #   ./serving_output/90389348723-23904872312/11_12_23_realistic.png
    #work_output_directory = os.path.join(SERVING_OUTPUT_DIRECTORY, work_id)
    os.makedirs(SERVING_OUTPUT_DIRECTORY, exist_ok=True)

    output_file_name = run_barber_process(
        BARBER_INPUT_DIRECTORY,
        input_file_name, style_file_name, color_file_name,
        'realistic', SERVING_OUTPUT_DIRECTORY
    )

    return jsonify({
        'message': 'Task is enqueued',
        'name': output_file_name
    })


# Usage:
# http://localhost:80/generated/21_45_24_realistic.png?work_id=8328723823-23912838232
# http://localhost:80/generated/832872382323912838232_45_24_realistic.png
@app.route('/generated/<path:path>')
#@jwt_required()  # jwt not upmost required for this, filename is equivalent to a token
def serve_outputs(path):
    #unsafe_work_id: str = request.args.get('work_id')
    #if unsafe_work_id is None:
        #return jsonify({'message': 'Parameter \'work_id\' is missing'})

    # Prevents path traversal vulnerability
    #safe_work_id = secure_filename(unsafe_work_id)

    #work_directory = os.path.join(OUTPUT_DIRECTORY, safe_work_id)

    #return send_from_directory(work_directory, path)

    return send_from_directory(SERVING_OUTPUT_DIRECTORY, path)


#@app.route('/api/barber/status', methods=['GET'])
#@jwt_required()
def api_barber_status():
    work_id: str = request.args.get('work_id')
    if work_id is None:
        return jsonify({'message': 'Parameter \'work_id\' is missing'}), 400

    #work: BarberTask = work_queue.get(work_id)
    #if work is None:
#        return jsonify({'message': 'work_id is invalid'}), 400

    # TODO return process status


#app.run(host='127.0.0.1', port=80)
app.run(host='127.0.0.1', port=443, ssl_context=('cert.pem', 'key.pem'))
#app.run(host='192.168.137.1', port=80)

#app.run(host='0.0.0.0', port=443, ssl_context=('cert.pem', 'key.pem'))
task_queue.join()
task_queue.put(None)  # signal exit
worker_thread.join()