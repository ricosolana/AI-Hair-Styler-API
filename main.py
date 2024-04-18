import collections
import json
import os
import queue
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
from enum import Enum, auto

import cv2
import numpy as np
import werkzeug.security
from flask import Flask, jsonify, request, make_response
from flask_jwt_extended import JWTManager, create_access_token, jwt_required
from werkzeug.utils import secure_filename
from winpty import PtyProcess

PROGRAM_START_TIME = time.perf_counter()


class TaskStatus(Enum):
    QUEUED = auto(), 'Queued'
    FACE_ALIGN = auto(), 'Face Align'
    BARBER_INIT = auto(), 'Initializing'
    EMBEDDING = auto(), 'Embedding'
    MASK_STEP1 = auto(), 'Mask Step1'
    MASK_STEP2 = auto(), 'Mask Step2'
    ALIGN_STEP1 = auto(), 'Align Step1'
    ALIGN_STEP2 = auto(), 'Align Step2'
    #MASK_STEP1_SECOND = auto(), 'Mask Step1 (2/2)'
    #MASK_STEP2_SECOND = auto(), 'Mask Step2 (2/2)'
    #ALIGN_STEP1_SECOND = auto(), 'Align Step1 (2/2)'
    #ALIGN_STEP2_SECOND = auto(), 'Align Step2 (2/2)'
    BLEND = auto(), 'Blend'
    COMPLETE = auto(), 'Complete'
    ERROR_FACE_ALIGN = auto(), 'Error with Face Align'  # image is too squished or no face detected (file never saved)
    ERROR_UNKNOWN = auto(), 'Unknown Error'
    ERROR_FATAL = auto(), 'Fatal Error'
    PROCESSING = auto(), 'Processing...'  # being crunched by barber (when we cannot track stdout)
    CANCELLED = auto(), 'Task Cancelled'


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

PATTERN_IMAGE_NUMBER = re.compile(r'Number of images: (\d+)')
#PATTERN_EMBEDDING_PROGRESS = re.compile(r'(\d+)\D*(\d+)\D*(\d+)\D*(\d+:\d+)<\??(\d+:\d+)\D*(\d+\.\d+)')
#(Embedding|Images|Create Target Mask Step\d*|Align Step \d*|Blend):\s*(\d+)\D*(\d+)\D*(\d+)\D*(\d+:\d+)<(\?|\d+:\d+),\s*(\?|\d+\.\d+)

# Standard names captured groups not supported in python, except if starting with 'P'
#   P?<name>
#(?<name>Embedding|Images|Create Target Mask Step\d*|Align Step \d*|Blend):\s*(?<percentage>\d+)\D*(?<numerator>\d+)\D*(?<denominator>\d+)\D*(?<actual>\d+:\d+)<(?<estimate>\?|\d+:\d+),\s*(?<rate>\?|\d+\.\d+)
# https://regexr.com/7v2uj
PATTERN_ANY_STAGE = re.compile(r'(?P<name>Embedding|Images|Create Target Mask Step\d*|Align Step \d*|Blend):\s*'
                               r'(?P<percentage>\d+)\D*(?P<numerator>\d+)\D*(?P<denominator>\d+)\D*'
                               r'(?P<actual>\d+:\d+)<(?P<estimate>\?|\d+:\d+),\s*(?P<rate>\?|\d+\.\d+)')

app.config["JWT_TOKEN_LOCATION"] = ["headers", "cookies"]
app.config['JWT_COOKIE_SECURE'] = True  # cookies over https only
app.config['JWT_SECRET_KEY'] = os.environ.setdefault('JWT_SECRET_KEY', secrets.token_urlsafe(32))
jwt = JWTManager(app)


def walk_single_file(_dir: str, name_pattern=None) -> str | None:
    pattern = re.compile(name_pattern) if name_pattern is not None else None
    with os.scandir(_dir) as entries:
        for entry in entries:
            if entry.is_file() and (name_pattern is None or pattern.search(entry.name)):
                return os.path.abspath(entry.path)
    return None


class StdDeque:
    def __init__(self, maxlen=32):
        if maxlen < 30:
            raise ValueError('maxlen must be at least 30 for reliable stdev')
        self._deque = collections.deque(maxlen=maxlen)

    def append(self, value: float):
        self._deque.append(value)

    def is_consistent(self, max_std=0.1):
        if len(self._deque) >= 30:
            std = np.std(self._deque)

            if std < max_std:
                return True

        return False

    def average(self):
        return np.average(self._deque)


task_queue = queue.Queue(maxsize=1000)
task_status_map = {}
task_status_lock = threading.Lock()


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

        self.status = TaskStatus.QUEUED
        #self.detailed_status = ''
        self.time_queued = time.perf_counter()  # point in seconds
        self.time_align_started = 0  # point in seconds
        self.time_barber_started = 0  # point in seconds
        self.time_barber_estimate = 0  # point in seconds
        self.time_barber_ended = 0  # point in seconds

        self.initial_barber_duration_estimate = 0  # duration in seconds

        self.current_transformer_percentage100 = None

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

    def _w_steps(self):
        return 1 if BARBERSHOP_FAST_GENERATION else max(1, int(1100.0 * self.quality))

    def _fs_steps(self):
        return 1 if BARBERSHOP_FAST_GENERATION else max(1, int(250.0 * self.quality))

    def _align_steps1(self):
        return 1 if BARBERSHOP_FAST_GENERATION else max(1, int(140.0 * self.quality))

    def _align_steps2(self):
        return 1 if BARBERSHOP_FAST_GENERATION else max(1, int(100.0 * self.quality))

    def _blend_steps(self):
        return 1 if BARBERSHOP_FAST_GENERATION else max(1, int(400.0 * self.quality))

    def _step_args(self):
        if self._is_fallback_barbershop():
            return []
        else:
            return [
                '--W_steps', str(self._w_steps()),
                '--FS_steps', str(self._fs_steps()),
                '--align_steps1', str(self._align_steps1()),
                '--align_steps2', str(self._align_steps2()),
                '--blend_steps', str(self._blend_steps())
            ]

    #def initial_barber_duration_estimate(self):
        #with self:
            #return self._initial_barbar_duration_estimate

#    def duration_barber(self):
#        with self:
#            return self.time_barber_ended - self.time_barber_started

    # time is converted to time.time()
#    def time_barber_started(self):
#        with self:
#            (self.duration_barber() - PROGRAM_START_TIME) + time.time()

    def set_status_concurrent(self, status: TaskStatus):
        with task_status_lock:
            self.status = status

    def execute(self):
        with task_status_lock:
            self.time_align_started = time.perf_counter()

        # Generate align output directory
        os.makedirs(self._abs_input_dir(), exist_ok=True)

        if not self._is_fallback_barbershop():
            self.set_status_concurrent(TaskStatus.FACE_ALIGN)

            align_proc = subprocess.run([
                sys.executable, BARBERSHOP_ALIGN,
                '-unprocessed_dir', self._abs_unprocessed_dir(),
                "-output_dir", self._abs_input_dir()
            ], env=os.environ, cwd=self._barbershop_dir())

            if align_proc.returncode != 0:
                self.set_status_concurrent(TaskStatus.ERROR_FACE_ALIGN)
                # error, we should note this
                return False
        else:
            shutil.copyfile(
                os.path.join(self._abs_unprocessed_dir(), self.unprocessed_face_path),
                os.path.join(self._abs_input_dir(), self.unprocessed_face_path))

        self.set_status_concurrent(TaskStatus.PROCESSING)

        if walk_single_file(self._abs_input_dir()) is None:
            self.set_status_concurrent(TaskStatus.ERROR_FACE_ALIGN)
            print('Image align did not recognize face; too squished?')
            return False

        # Generate barber output directory
        os.makedirs(os.path.join(self._abs_output_dir(), 'W+'), exist_ok=True)
        os.makedirs(os.path.join(self._abs_output_dir(), 'FS'), exist_ok=True)
        os.makedirs(os.path.join(self._abs_output_dir(), 'Blend_realistic'), exist_ok=True)
        os.makedirs(os.path.join(self._abs_output_dir(), 'Align_realistic'), exist_ok=True)

        _proc_buffered_line = []

        def readline(proc: PtyProcess, ctrl_chars=True):
            while 1:
                try:
                    ch = proc.read(1)
                except EOFError:
                    return ''.join(_proc_buffered_line)

                if ch:
                    if ch in ('\n', '\r'):
                        out = ''.join(_proc_buffered_line)
                        _proc_buffered_line.clear()
                        return out

                    if ctrl_chars or ord(ch) >= 0x20:
                        _proc_buffered_line.append(ch)

                return ''

        with task_status_lock:
            self.time_barber_started = time.perf_counter()

        args = [
                   sys.executable, self._barbershop_main(),
                   '--input_dir', self._abs_input_dir(),
                   "--im_path1", self._walk_rel_input_face_file(),  # face
                   "--im_path2", self._rel_input_style_file(),  # style
                   "--im_path3", self._rel_input_color_file(),  # color
                   "--sign", 'realistic',
                   "--smooth", '5',

                   "--output_dir", self._abs_output_dir()  # work_output_directory
               ] + self._step_args()

        barber_proc = PtyProcess.spawn(
            ' '.join(args),
            cwd=self._barbershop_dir(),
            env=os.environ)

        def _watcher():
            while barber_proc.isalive():
                if barber_proc.exitstatus is not None:
                    try:
                        barber_proc.close()
                    finally:
                        break

                time.sleep(0.01)

        barber_proc_watchdog = threading.Thread(target=_watcher, name='BarberTaskWatchdog')
        barber_proc_watchdog.start()

        current_image_number = 0
        current_transformer_stage = ''
        #current_transformer_percentage100 = 0
        std_deque = StdDeque()
        image_count = 0
        skip_stats = self.quality < 0.03 or self._is_fallback_barbershop()
        initial_barbar_duration_estimate = 0
        mask_or_align_iteration = 1
        #detailed_status = ''
        last_status_detail_update_time = time.perf_counter()

        print(f'skipping stats: {skip_stats}')

        self.set_status_concurrent(TaskStatus.BARBER_INIT)

        while 1:
            try:
                line = readline(barber_proc, ctrl_chars=False)
            except (EOFError, ConnectionAbortedError):
                break

            if barber_proc.exitstatus is not None:
                break

            if line and not skip_stats:
                #print(line)

                """
                Image count extraction (once)
                """
                if image_count == 0:
                    match_image_count = PATTERN_IMAGE_NUMBER.search(line)
                    if match_image_count is not None:
                        image_count = int(match_image_count.group(1))

                """
                Embedded matcher
                """
                match_any_stage = PATTERN_ANY_STAGE.search(line)
                if match_any_stage is not None:
                    """
                    Current stage info extractor
                    """
                    # little optimization to not lock every print
                    _current_transformer_stage = match_any_stage.group('name')
                    _current_transformer_percentage100 = int(match_any_stage.group('percentage'))
                    if _current_transformer_stage != current_transformer_stage:
                        current_transformer_stage = _current_transformer_stage

                        print('updated status')

                        with task_status_lock:
                            # immediately updates for outdated periodic check
                            self.current_transformer_percentage100 = 0
                            match current_transformer_stage:
                                case 'Embedding':
                                    self.status = TaskStatus.EMBEDDING
                                case 'Create Target Mask Step1':
                                    mask_or_align_iteration += 1
                                    self.status = TaskStatus.MASK_STEP1
                                case 'Create Target Mask Step2':
                                    self.status = TaskStatus.MASK_STEP2
                                case 'Align Step 1':
                                    self.status = TaskStatus.ALIGN_STEP1
                                case 'Align Step 2':
                                    self.status = TaskStatus.ALIGN_STEP2
                                case 'Blend':
                                    self.status = TaskStatus.BLEND
                                case _:
                                    pass

                    # TODO create an updater for this
                    #current_transformer_percentage100 = _current_transformer_percentage100

                    now = time.perf_counter()
                    # update status detail every few seconds or so
                    if now - last_status_detail_update_time >= 1:
                        last_status_detail_update_time = now
                        # update detail
                        # TODO is lock necessary here
                        with task_status_lock:
                            self.current_transformer_percentage100 = _current_transformer_percentage100
                            #self.detailed_status = '%s' % self.status.value[1]

                    """
                    TODO 'Images' extractor
                    """
                    """
                    if current_transformer_stage == 'Images':
                        _current_image_number = match_any_stage.group('numerator')
                        if current_image_number != _current_image_number:
                            current_image_number = _current_image_number
                            # TODO update embedded iteration string and progress
                            #detailed_status = '%s' % 'h'
                    """



                    """
                    Time predictor (once)
                    """
                    if initial_barbar_duration_estimate == 0:
                        rate_str = match_any_stage.group('rate')
                        if rate_str != '?':
                            val = float(rate_str)
                            std_deque.append(val)
                            if std_deque.is_consistent(0.05):
                                reliable_it_s = std_deque.average()

                                # calculate estimate time for entire process:
                                total_steps = (self._w_steps() + self._fs_steps()) * image_count \
                                            + (self._align_steps1() + self._align_steps2()) * 2 + self._blend_steps()

                                # target mask (step1 / step2 ) both times
                                total_steps += 80 * 4

                                # subtract current step
                                total_steps -= int(match_any_stage.group('numerator'))

                                initial_barbar_duration_estimate = total_steps / reliable_it_s

                                with task_status_lock:
                                    self.time_barber_estimate = time.perf_counter()
                                    self.initial_barber_duration_estimate = initial_barbar_duration_estimate

                                print(f'Initial time estimate: {initial_barbar_duration_estimate}s')
            else:
                time.sleep(0)

        barber_proc_watchdog.join()

        if walk_single_file(self._abs_output_dir(), '^image') is not None:
            self.set_status_concurrent(TaskStatus.COMPLETE)
            return True

        self.set_status_concurrent(TaskStatus.ERROR_UNKNOWN)
        return False


def worker():
    while True:
        # blocks until item available, pops it
        task_current: CompiledProcess = task_queue.get()

        print(f'Processing {task_current.work_id}')

        start_time = time.perf_counter()
        #success = False
        #try:
        success = task_current.execute()
        #except Exception:
            #print(traceback.format_exc())
            #task_current.set_status_concurrent(TaskStatus.ERROR_FATAL)

        end_time = time.perf_counter()

        print(f'Task {task_current.work_id} '
              f'{"succeeded" if success else "failed"} after {end_time - start_time} seconds')

        # lock
        with task_status_lock:
            task_current.time_barber_ended = end_time

        task_queue.task_done()


worker_thread = threading.Thread(target=worker, name='BarberTaskWorker')
worker_thread.start()


def response_safe_serve_image(directory, unsafe_path):
    return response_unsafe_serve_image(werkzeug.security.safe_join(directory, unsafe_path))


def response_unsafe_serve_image(safe_path, width=None, quality=90):
    image = cv2.imread(safe_path)
    if image is None:
        return jsonify({'message': 'Image not found'}), 400

    img_height, img_width, _ = image.shape
    #width = min(img_width, width if width is not None else)
    #if img_width > 600:
        #width =
    if width is not None or img_width > 600:
        # If image is just too big
        width = min(256, width if width is not None else img_width)

        # Sanitize width [1, img_width]
        width = max(1, min(width, img_width))

        dsize = (width, int(img_width * (width / img_height)))

        image = cv2.resize(image, dsize=dsize)
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
    task = CompiledProcess(work_id,
                           unprocessed_face_path,
                           style_filename,
                           color_filename,
                           demo=demo,
                           quality=quality)
    with task_status_lock:
        # CompiledProcess.task_current = task_current
        task_status_map[task.work_id] = task

    task_queue.put(task)


# Root path
@app.route('/', methods=['GET'])
def index_path():
    return jsonify({
        'name': 'ai hair styler generator api',
        #'task-queue': task_queue.qsize(),
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

    width = request.args.get('width')
    quality = request.args.get('quality', '90')

    if width is not None:
        try:
            width = int(width)
        except (ValueError, TypeError):
            return jsonify({'message': 'Width must be an integer'}), 400

    try:
        quality = int(quality)
    except ValueError:
        return jsonify({'message': 'Quality must be an integer'}), 400

    if os.path.exists(safe_path):
        # TODO replace with above walk_single_file with regex for ^image
        with os.scandir(safe_path) as entries:
            for entry in entries:
                # ^([0-9a-f]{64})_(.+)_(.+)_(\w+).png
                if entry.is_file() and entry.name.startswith('image'):
                    return response_unsafe_serve_image(entry.path, width=width, quality=quality)

    return jsonify({'message': 'Image not found'}), 400


@app.route('/api/status', methods=['GET'])
@jwt_required()
def api_status():
    work_id = request.args.get('work-id')
    if work_id is None:
        return jsonify({'message': 'Parameter \'work-id\' is missing'}), 400

    with task_status_lock:
        task: CompiledProcess = task_status_map.get(work_id)

        if not task:
            return jsonify({'message': 'Task not found'}), 400

        duration_barber = task.time_barber_ended - task.time_barber_started
        estimate = task.initial_barber_duration_estimate

        # time.time()

        js = {
            # TODO optionally include this without the name
            #'status': task.status.name,
            'status': task.status.name,  # raw status value
            'status-label': task.status.value[1],  # status readeable
            'current-transformer-percentage': task.current_transformer_percentage100,
            #'status-value': task.status.value,
            'time-queued': task.time_queued,
            'time-align-started': task.time_align_started,
            'time-barber-started': task.time_barber_started,
            #'time-barber-started': task.time_barber_started(),
            'time-barber-ended': task.time_barber_ended,
            'initial-barber-duration-estimate': estimate,
            #'initial-barber-duration-estimate': task.initial_barber_duration_estimate(),
            #'duration-estimate-difference': ((estimate - duration_barber) / estimate) if estimate != 0 else 0,
            'duration-barber': duration_barber,
            #'duration-barber': task.duration_barber()
            #'utc-estimated-end'
        }

        return jsonify(js)


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
