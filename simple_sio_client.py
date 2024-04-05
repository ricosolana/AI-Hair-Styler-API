import socketio

sio = socketio.Client(logger=True, engineio_logger=True)

@sio.event
def response_on_converted(status, data):
    print("Converted response")

sio.connect(url='http://127.0.0.1:80')

with open('test_partial_side.jpg', 'rb') as f:
    sio.emit('/api/svoji', f.read(), callback=response_on_converted)
