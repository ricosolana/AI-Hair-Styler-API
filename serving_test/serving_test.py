from flask import Flask, send_from_directory
import os

fake = 'my/file/is/here/pic.jpg'
split = os.path.splitext(fake)

app = Flask(__name__)


@app.route('/reports/<path:path>')
def api_barber_poll(path):
    return send_from_directory('reports', path)


app.run(host='127.0.0.1', port=80)
