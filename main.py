import json
import os
import secrets
import subprocess
import queue
import sys
import threading
import cv2
import numpy as np
import werkzeug.security

from flask import Flask, jsonify, request, abort, send_from_directory, make_response
from flask_jwt_extended import JWTManager, create_access_token, jwt_required
from werkzeug.utils import secure_filename

# hmm idk
# openssl req -x509 -newkey rsa:4096 -nodes -out cert.pem -keyout key.pem -days 365


app = Flask(__name__)

if not app.config.from_file('config.json', load=json.load):
    print('Failed to load config')
    exit(1)

BARBER_MAIN = app.config['BARBER_MAIN']
FAKE_BARBER_MAIN = app.config['FAKE_BARBER_MAIN']
BARBER_FACES_INPUT_DIRECTORY = app.config['BARBER_FACES_INPUT_DIRECTORY']
SERVING_PROCESSED_OUTPUT_DIRECTORY = app.config['SERVING_PROCESSED_OUTPUT_DIRECTORY']
SERVING_TEMPLATE_INPUT_DIRECTORY = app.config['SERVING_TEMPLATE_INPUT_DIRECTORY']

TEMPLATE_DIRECTORY_FILE_LIST = [f for f in os.listdir(SERVING_TEMPLATE_INPUT_DIRECTORY)
                                if os.path.isfile(os.path.join(SERVING_TEMPLATE_INPUT_DIRECTORY, f))]



#rel_template_input_directory = os.path.relpath(path=os.path.abspath(BARBER_FACES_INPUT_DIRECTORY))

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

        subprocess.run(task, env=os.environ)
        task_queue.task_done()


worker_thread = threading.Thread(target=worker)
worker_thread.start()


# This endpoint is accessible only from localhost
@app.route('/', methods=['GET'])
def index_path():
    return jsonify({'name': 'ai hair styler generator api',
                    #'message': '/auth/token, /api/barber, /generated/708dd2bab3b011676bb80d640f363838c5754f8000cc67b9503072fc6b1e96f9_12_123_realistic.png',
                    'task-queue': task_queue.qsize(),
                    'version': 'v1.1.0'
                    })


# This endpoint is accessible only from localhost
@app.route('/auth/token', methods=['GET'])
def api_token():
    if request.remote_addr != '127.0.0.1':
        abort(403)  # Forbidden

    # Generate a token without requiring user_id or any other data
    access_token = create_access_token(identity="anonymous")
    print(f'Generated token: {access_token}')
    return jsonify(access_token=access_token)


def run_barber_process(_barber_main, input_dir, im_path1, im_path2, im_path3, sign, output_dir):
    task_queue.put([
        #"python", _barber_main, #BARBER_MAIN,
        sys.executable, _barber_main,
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

    style_file_name = secure_filename(request.args.get('style-file'))
    if style_file_name is None:
        return jsonify({'message': 'Parameter \'style-file\' is missing'}), 400

    color_file_name = secure_filename(request.args.get('color-file'))
    if color_file_name is None:
        return jsonify({'message': 'Parameter \'color-file\' is missing'}), 400

    work_id = secrets.token_hex(32)

    ext = os.path.splitext(f.filename)[-1].lower()
    if ext not in ('.png', '.jpg', '.jpeg'):
        return jsonify({'message': f'Image extension \'{ext}\' is invalid'})

    #rel_input_face_file = work_id + ext

    # Read the image file into a numpy array
    # Decode the image using OpenCV
    cv_image = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
    if cv_image is None:
        return jsonify({'message': 'Image parsing failed'}), 400

    #cv_image = cv2.imread(os.path.abspath(im_path1))

    face_file_input_name = work_id + ext
    os.makedirs(BARBER_FACES_INPUT_DIRECTORY, exist_ok=True)
    f.stream.seek(0)
    f.save(os.path.join(BARBER_FACES_INPUT_DIRECTORY, face_file_input_name))
    #cv2.imwrite(os.path.join(BARBER_FACES_INPUT_DIRECTORY, face_file_input_name), cv_image)

    # Served image is physically saved to
    #   ./serving_output/90389348723-23904872312/11_12_23_realistic.png
    #work_output_directory = os.path.join(SERVING_OUTPUT_DIRECTORY, work_id)
    os.makedirs(SERVING_PROCESSED_OUTPUT_DIRECTORY, exist_ok=True)

    # color_file_name
    # os.path.join(SERVING_STYLE_INPUT_DIRECTORY, style_file_name)

    # the face/style/color images are normally in the same directory
    # we want to separate faces/style/color from each other, into different directories,
    # so use relative paths

    # determine the relative path for:
    #   input/face/43.png
    abs_input_directory = os.path.abspath(SERVING_TEMPLATE_INPUT_DIRECTORY)
    rel_face_input_directory = os.path.relpath(os.path.abspath(BARBER_FACES_INPUT_DIRECTORY), abs_input_directory)
    rel_template_input_directory = os.path.relpath(os.path.abspath(SERVING_TEMPLATE_INPUT_DIRECTORY),
                                                   abs_input_directory)

    rel_face_file = os.path.join(rel_face_input_directory, face_file_input_name)
    rel_style_file = os.path.join(rel_template_input_directory, style_file_name)
    rel_color_file = os.path.join(rel_template_input_directory, color_file_name)

    # the style/color path are relative to the
    # BARBER_FACES_INPUT_DIRECTORY

    output_file_name = run_barber_process(
        FAKE_BARBER_MAIN if request.args.get('demo', '0') == '1' else BARBER_MAIN,
        abs_input_directory,
        rel_face_file,
        rel_style_file,
        rel_color_file,
        'realistic', SERVING_PROCESSED_OUTPUT_DIRECTORY
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
def serve_generated(path):
    #unsafe_work_id: str = request.args.get('work_id')
    #if unsafe_work_id is None:
        #return jsonify({'message': 'Parameter \'work_id\' is missing'})

    # Prevents path traversal vulnerability
    #safe_work_id = secure_filename(unsafe_work_id)

    #work_directory = os.path.join(OUTPUT_DIRECTORY, safe_work_id)

    #return send_from_directory(work_directory, path)

    return response_compressed(SERVING_PROCESSED_OUTPUT_DIRECTORY, path)

    #return send_from_directory(SERVING_PROCESSED_OUTPUT_DIRECTORY, path)


@app.route('/api/templates/styles', methods=['GET'])
#@jwt_required()
def api_templates_styles():
    return jsonify(app.config['STYLES'])


@app.route('/api/templates/list', methods=['GET'])
#@jwt_required()
def api_templates_list():
    # dont want to iterate all files
    return jsonify(TEMPLATE_DIRECTORY_FILE_LIST)


@app.route('/api/templates/colors', methods=['GET'])
#@jwt_required()
def api_templates_colors():
    return jsonify(app.config['COLORS'])


def response_compressed(directory, path):
    path = secure_filename(path)
    # werkzeug.security.safe_join()
    image = cv2.imread(os.path.join(directory, path))
    if image is None:
        return jsonify({'message': 'Image not found'}), 400

    image = cv2.resize(image, (256, 256))
    success, arr = cv2.imencode('.jpg', image,
                                [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    response = make_response(arr.tobytes())
    response.headers.set('Content-Type', 'image/jpeg')

    return response


@app.route('/templates/<path:path>', methods=['GET'])
#@jwt_required()
def serve_templates(path):
    return response_compressed(SERVING_TEMPLATE_INPUT_DIRECTORY, path)
    """
    path = secure_filename(path)
    #werkzeug.security.safe_join()
    image = cv2.imread(os.path.join(SERVING_TEMPLATE_INPUT_DIRECTORY, path))
    if image is None:
        return jsonify({'message': 'Image not found'}), 400

    image = cv2.resize(image, (256, 256))
    success, arr = cv2.imencode('.jpg', image,
                                [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    response = make_response(arr.tobytes())
    response.headers.set('Content-Type', 'image/jpeg')

    return response
    #return send_from_directory(SERVING_TEMPLATE_INPUT_DIRECTORY, path)
    """


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


app.run(host='127.0.0.1', port=80)
#app.run(host='127.0.0.1', port=443, ssl_context=('certs1/server-cert.pem', 'certs1/server-key.pem'))
#app.run(host='192.168.137.1', port=80)

#app.run(host='0.0.0.0', port=443, ssl_context=('certs1/server-cert.pem', 'certs1/server-key.pem'))
task_queue.join()
task_queue.put(None)  # signal exit
worker_thread.join()