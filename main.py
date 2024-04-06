import json
import os
import secrets
import subprocess

from flask import Flask, jsonify, request, abort, send_from_directory
from flask_jwt_extended import JWTManager, create_access_token, jwt_required
from werkzeug.utils import secure_filename

# openssl req -x509 -newkey rsa:4096 -nodes -out cert.pem -keyout key.pem -days 365

app = Flask(__name__)

if not app.config.from_file('config.json', load=json.load):
    print('Failed to load config')
    exit(1)

INPUT_DIRECTORY = app.config['INPUT_DIRECTORY']
OUTPUT_DIRECTORY = app.config['OUTPUT_DIRECTORY']

os.environ.setdefault('aihairstyler_JWT_SECRET_KEY', secrets.token_urlsafe(32))
app.config['JWT_SECRET_KEY'] = os.environ['aihairstyler_JWT_SECRET_KEY']

jwt = JWTManager(app)

work_queue = {}


class WorkStatus:
    def __init__(self, work_id: str, process: subprocess.Popen):
        self.work_id = work_id
        self.process = process


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

    style_file_name = app.config['STYLES'].get(style)
    if style_file_name is None:
        return jsonify({'message': f'Hairstyle \'{style}\' is invalid'}), 400

    color_file_name = app.config['COLORS'].get(color)
    if color_file_name is None:
        return jsonify({'message': f'Hair color \'{color}\' is invalid'}), 400

    work_id = secrets.token_urlsafe(32)

    ext = secure_filename(os.path.splitext(f.filename)[-1].lower())
    if ext not in ('.png', '.jpg', '.jpeg'):
        return jsonify({'message': f'Image extension \'{ext}\' is invalid'})

    input_file_name = work_id + ext
    os.makedirs(INPUT_DIRECTORY)
    f.save(os.path.join(INPUT_DIRECTORY, input_file_name))

    # Served image is physically saved to
    #   ./serving_output/90389348723-23904872312/11_12_23_realistic.png
    work_output_directory = os.path.join(OUTPUT_DIRECTORY, work_id)
    os.makedirs(work_output_directory)

    process = subprocess.Popen([
        "python", "barber.py",
        '--input_dir', INPUT_DIRECTORY,
        "--im_path1", input_file_name,  # face
        "--im_path2", style_file_name,  # style
        "--im_path3", color_file_name,  # color
        "--sign", "realistic",
        "--smooth", "5",
        "--output_dir", work_output_directory
    ])

    work_queue[work_id] = WorkStatus(work_id, process)

    # TODO ensure this name matches the outputted file from barbershop
    output_file_name = (f'{os.path.splitext(os.path.basename(input_file_name))[0]}_'
                        f'{style}_{color}_realistic.png')

    return jsonify({
        'message': 'Image is being processed',
        'work_id': work_id,
        'name': output_file_name
    })


# Usage:
# http://localhost:80/generated/21_45_24_realistic.png?work_id=8328723823-23912838232
@app.route('/generated/<path:path>')
@jwt_required()
def serve_outputs(path):
    unsafe_work_id: str = request.args.get('work_id')
    if unsafe_work_id is None:
        return jsonify({'message': 'Parameter \'work_id\' is missing'})

    # Prevents path traversal vulnerability
    safe_work_id = secure_filename(unsafe_work_id)

    work_directory = os.path.join(OUTPUT_DIRECTORY, safe_work_id)

    return send_from_directory(work_directory, path)


#@app.route('/api/barber/status', methods=['GET'])
#@jwt_required()
def api_barber_status():
    work_id: str = request.args.get('work_id')
    if work_id is None:
        return jsonify({'message': 'Parameter \'work_id\' is missing'}), 400

    work: WorkStatus = work_queue.get(work_id)
    if work is None:
        return jsonify({'message': 'work_id is invalid'}), 400

    # TODO return process status


#app.run(host='127.0.0.1', port=80)
app.run(host='127.0.0.1', port=443, ssl_context=('cert.pem', 'key.pem'))
#app.run(host='192.168.137.1', port=80)

#app.run(port=443, ssl_context=('cert.pem', 'key.pem'))