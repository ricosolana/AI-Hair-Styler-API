import json
import os
import queue
import secrets
import shutil
import subprocess
import sys
import threading
import time

import cv2
import numpy as np
import werkzeug.security
from flask import Flask, jsonify, request, make_response
from flask_jwt_extended import JWTManager, create_access_token, jwt_required
from werkzeug.utils import secure_filename


app = Flask(__name__)

if not app.config.from_file('config.json', load=json.load):
    print('Failed to load config')
    exit(1)

BARBERSHOP_DIR = app.config['BARBERSHOP_DIR']
BARBERSHOP_FAST_GENERATION = app.config['BARBERSHOP_FAST_GENERATION']
USE_FALLBACK_BARBERSHOP = app.config['USE_FALLBACK_BARBERSHOP']
FALLBACK_BARBERSHOP_MAIN = app.config['FALLBACK_BARBERSHOP_MAIN']
FACES_UNPROCESSED_INPUT_DIRECTORY = app.config['FACES_UNPROCESSED_INPUT_DIRECTORY']
FACES_INPUT_DIRECTORY = app.config['FACES_INPUT_DIRECTORY']
SERVING_OUTPUT_DIRECTORY = app.config['SERVING_PROCESSED_OUTPUT_DIRECTORY']

BARBERSHOP_MAIN = os.path.join(BARBERSHOP_DIR, 'main.py')
BARBERSHOP_ALIGN = os.path.join(BARBERSHOP_DIR, 'align_face.py')
SERVING_TEMPLATE_INPUT_DIRECTORY = os.path.join(BARBERSHOP_DIR, 'input', 'face')

TEMPLATE_DIRECTORY_FILE_LIST = [f for f in os.listdir(SERVING_TEMPLATE_INPUT_DIRECTORY)
                                if os.path.isfile(os.path.join(SERVING_TEMPLATE_INPUT_DIRECTORY, f))]



app.config["JWT_TOKEN_LOCATION"] = ["headers", "cookies"]
app.config['JWT_COOKIE_SECURE'] = True  # cookies over https only
app.config['JWT_SECRET_KEY'] = os.environ.setdefault('JWT_SECRET_KEY', secrets.token_urlsafe(32))
jwt = JWTManager(app)

# thread-safe
# TODO use PriorityQueue to prioritize certain requests
#   maybe VIP or developer tasks
task_queue = queue.Queue(maxsize=1000)


def walk_single_file(_dir: str | os.PathLike) -> str | None:
    with os.scandir(_dir) as entries:
        for entry in entries:
            if entry.is_file():
                return os.path.abspath(entry.path)
    return None


class CompiledProcess:
    def __init__(self,
                 work_id: str,
                 unprocessed_face_path: str | os.PathLike,
                 style_filename: str,
                 color_filename: str,
                 demo: bool,
                 quality: float):
        self.work_id = work_id
        self.unprocessed_face_path = unprocessed_face_path
        self.style_filename = style_filename
        self.color_filename = color_filename
        self.demo = demo
        self.quality = quality

    def get_progress(self):
        # somehow after reading from stdout, retrieve some
        # stats to get some good estimations on where we are
        pass

    # Constant
    def _abs_unprocessed_dir(self):
        return os.path.abspath(os.path.join(FACES_UNPROCESSED_INPUT_DIRECTORY,
                                            self.work_id))

    # Constant
    def _abs_input_dir(self):
        return os.path.abspath(os.path.join(FACES_INPUT_DIRECTORY,
                                            self.work_id))

    def _abs_template_dir(self):
        return os.path.abspath(SERVING_TEMPLATE_INPUT_DIRECTORY)

    def _rel_template_dir(self):
        return os.path.relpath(self._abs_template_dir(), self._abs_input_dir())

    # Walks over disk to extract name from single-file
    def _walk_rel_input_face_file(self):
        # no file found,
        abs_input_face_file = walk_single_file(self._abs_input_dir())
        return os.path.relpath(abs_input_face_file, self._abs_input_dir())

    def _rel_input_style_file(self):
        return os.path.join(self._rel_template_dir(), self.style_filename)

    def _rel_input_color_file(self):
        return os.path.join(self._rel_template_dir(), self.color_filename)

    def _abs_output_dir(self):
        return os.path.join(os.path.abspath(SERVING_OUTPUT_DIRECTORY), self.work_id)

    def _is_fallback_barbershop(self):
        return self.demo or USE_FALLBACK_BARBERSHOP

    def _barbershop_main(self):
        return FALLBACK_BARBERSHOP_MAIN if self._is_fallback_barbershop() else BARBERSHOP_MAIN

    def _barbershop_dir(self):
        return os.path.abspath('.') if self._is_fallback_barbershop() else BARBERSHOP_DIR

    def _step_args(self):
        if self._is_fallback_barbershop():
            return []
        else:
            if BARBERSHOP_FAST_GENERATION:
                return [
                    '--W_steps', '1',
                    '--FS_steps', '1',
                    '--align_steps1', '1',
                    '--align_steps2', '1',
                    '--blend_steps', '1'
                ]
            else:
                return [
                    '--W_steps', f'{max(1, int(1100.0 * self.quality))}',
                    '--FS_steps', f'{max(1, int(250.0 * self.quality))}',
                    '--align_steps1', f'{max(1, int(140.0 * self.quality))}',
                    '--align_steps2', f'{max(1, int(100.0 * self.quality))}',
                    '--blend_steps', f'{max(1, int(400.0 * self.quality))}'
                ]

    def execute(self):
        # Generate align output directory
        os.makedirs(self._abs_input_dir(), exist_ok=True)

        if not self._is_fallback_barbershop():
            align_proc = subprocess.run([
                sys.executable, BARBERSHOP_ALIGN,
                '-unprocessed_dir', self._abs_unprocessed_dir(),
                "-output_dir", self._abs_input_dir()
            ], env=os.environ, cwd=self._barbershop_dir())

            # alternatively, check that the file was actually generated
            #   this is the ultimate best condition

            # in what cases would image not be generated?

            if align_proc.returncode != 0:
                # error, we should note this
                return False
        else:
            shutil.copyfile(
                os.path.join(self._abs_unprocessed_dir(), self.unprocessed_face_path),
                os.path.join(self._abs_input_dir(), self.unprocessed_face_path))

        if walk_single_file(self._abs_input_dir()) is None:
            print('Image align did not recognize face; too squished?')
            return False

        # Generate barber output directory
        os.makedirs(os.path.join(self._abs_output_dir(), 'W+'), exist_ok=True)
        os.makedirs(os.path.join(self._abs_output_dir(), 'FS'), exist_ok=True)
        os.makedirs(os.path.join(self._abs_output_dir(), 'Blend_realistic'), exist_ok=True)
        os.makedirs(os.path.join(self._abs_output_dir(), 'Align_realistic'), exist_ok=True)

        barber_proc = subprocess.run([
            sys.executable, self._barbershop_main(),
            '--input_dir', self._abs_input_dir(),
            "--im_path1", self._walk_rel_input_face_file(),  # face
            "--im_path2", self._rel_input_style_file(),  # style
            "--im_path3", self._rel_input_color_file(),  # color
            "--sign", 'realistic',
            "--smooth", '5',

            "--output_dir", self._abs_output_dir()  # work_output_directory
        ] + self._step_args(), env=os.environ, cwd=self._barbershop_dir())

        if barber_proc.returncode != 0:
            return False

        return True


def worker():
    while True:
        # blocks until item available, pops it
        task: CompiledProcess = task_queue.get()

        print(f'Processing {task.work_id}')

        start_time = time.perf_counter()
        success = False
        try:
            success = task.execute()
        except Exception as err:
            print(err)
            print(f'Error: {type(err)}')

        end_time = time.perf_counter()
        print(f'Task {task.work_id} {"succeeded" if success else "failed"} after {end_time - start_time} seconds')

        task_queue.task_done()


worker_thread = threading.Thread(target=worker, name='BarberTaskWorker')
worker_thread.start()


def response_safe_serve_image(directory, unsafe_path):
    return response_unsafe_serve_image(werkzeug.security.safe_join(directory, unsafe_path))


def response_unsafe_serve_image(safe_path, width=None, quality=90):
    image = cv2.imread(safe_path)
    if image is None:
        return jsonify({'message': 'Image not found'}), 400

    if width is not None:
        img_height, img_width, _ = image.shape
        image = cv2.resize(image, (width, img_width * (width / img_height)))
    success, arr = cv2.imencode('.jpg', image,
                                [int(cv2.IMWRITE_JPEG_QUALITY), quality])

    response = make_response(arr.tobytes())
    response.headers.set('Content-Type', 'image/jpeg')

    return response


def compile_process(work_id: str,
                    unprocessed_face_path: str | os.PathLike,
                    style_filename: str, color_filename: str,
                    demo: bool,
                    quality: float):
    task_queue.put(CompiledProcess(work_id,
                                   unprocessed_face_path,
                                   style_filename,
                                   color_filename,
                                   demo=demo,
                                   quality=quality))


# Root path
@app.route('/', methods=['GET'])
def index_path():
    return jsonify({
        'name': 'ai hair styler generator api',
        'task-queue': task_queue.qsize(),
        'version': 'v1.2.0'
    })


# This endpoint is accessible only from localhost
@app.route('/auth/token', methods=['GET'])
def api_auth_token():
    if request.remote_addr != '127.0.0.1' or request.headers['Host'] != 'localhost':
        return jsonify({'message': 'Must be localhost'}), 403  # Forbidden

    # Generate a token without requiring user_id or any other data
    access_token = create_access_token(identity="anonymous")
    print(f'Generated token: {access_token}')
    return jsonify(access_token=access_token)


@app.route('/auth/check', methods=['GET'])
@jwt_required()
def api_auth_check():
    return jsonify({'message': 'Success'})


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

    quality_text = request.args.get('quality', '1.0')
    try:
        quality = float(quality_text)
    except ValueError:
        return jsonify({'message': 'Parameter \'quality\' must be a float'}), 400

    quality = max(0, min(quality, 1))

    work_id = secrets.token_hex(32)

    ext = os.path.splitext(f.filename)[-1].lower()
    if ext not in ('.png', '.jpg', '.jpeg'):
        return jsonify({'message': f'Image extension \'{ext}\' is invalid'})

    # Validate image
    cv_image = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
    if cv_image is None:
        return jsonify({'message': 'Image parsing failed'}), 400

    # Save the image to align input directory
    unprocessed_file_name = 'image' + ext
    unprocessed_input_dir = os.path.join(FACES_UNPROCESSED_INPUT_DIRECTORY, work_id)
    os.makedirs(unprocessed_input_dir, exist_ok=True)
    f.stream.seek(0)

    h, w, _ = cv_image.shape
    image_is_uneven = h != w
    if image_is_uneven:
        # resize image to fit
        cv_image = cv2.resize(cv_image, (min(w, h), min(w, h)))

    cv2.imwrite(os.path.join(unprocessed_input_dir, unprocessed_file_name), cv_image)

    #f.save(os.path.join(unprocessed_input_dir, unprocessed_file_name))

    demo = request.args.get('demo', 'false') == 'true'

    compile_process(work_id, unprocessed_file_name, style_file_name, color_file_name, demo, quality)

    return jsonify({
        'message': 'Task is enqueued',
        'work-id': work_id
    })


# Usage:
#   http://localhost:80/generated/6b3a1377639e398ed126b32aafaa73c73ce03d9f75abddafc0d9feedbdcaafe3
@app.route('/generated/<path:path>')
#@jwt_required()
def serve_generated(path):
    safe_path = werkzeug.security.safe_join(SERVING_OUTPUT_DIRECTORY, path)

    if os.path.exists(safe_path):
        with os.scandir(safe_path) as entries:
            for entry in entries:
                # ^([0-9a-f]{64})_(.+)_(.+)_(\w+).png
                if entry.is_file() and entry.name.startswith('image'):
                    return response_unsafe_serve_image(entry.path)

    return jsonify({'message': 'Image not found'}), 400


@app.route('/api/templates/styles', methods=['GET'])
def api_templates_styles():
    return jsonify(app.config['STYLES'])


@app.route('/api/templates/list', methods=['GET'])
def api_templates_list():
    return jsonify(TEMPLATE_DIRECTORY_FILE_LIST)


@app.route('/api/templates/colors', methods=['GET'])
def api_templates_colors():
    return jsonify(app.config['COLORS'])


@app.route('/templates/<path:path>', methods=['GET'])
def serve_templates(path):
    return response_safe_serve_image(SERVING_TEMPLATE_INPUT_DIRECTORY, path)


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
"""

app.run(host='127.0.0.1', port=80)
#app.run(host='192.168.137.1', port=80)
