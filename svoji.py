# I installed GTK https://github.com/Kozea/CairoSVG/issues/371#issuecomment-1539570439
# I tried the below dll searcher/importer
# I manually installed cairo using vcpkg
#   if/when that fails, make sure your working directory path has NO SPACES

import os

def set_dll_search_path():
   # Python 3.8 no longer searches for DLLs in PATH, so we have to add
   # everything in PATH manually. Note that unlike PATH add_dll_directory
   # has no defined order, so if there are two cairo DLLs in PATH we
   # might get a random one.
   if os.name != "nt" or not hasattr(os, "add_dll_directory"):
       return
   for p in os.environ.get("PATH", "").split(os.pathsep):
       try:
           os.add_dll_directory(p)
       except OSError:
           pass

# because cairo and python3.8 is fucking stupid
set_dll_search_path()


from io import BytesIO
import cv2
import mediapipe as mp
import numpy as np
import time
import drawsvg

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)


def compile_connected(d: drawsvg.Drawing, face_landmarks, indices):
    p = drawsvg.Path(stroke_width=2, stroke='lime', fill='black', fill_opacity=0.2)

    # first point
    p.M(face_landmarks.landmark[indices[0]].x * d.width, face_landmarks.landmark[indices[0]].y * d.height)

    # remaining points
    for i in indices[1:]:
        p.L(face_landmarks.landmark[i].x * d.width, face_landmarks.landmark[i].y * d.height)

    d.append(p)


def process_svoji(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(image)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            d = drawsvg.Drawing(800, 400)

            compile_connected(d, face_landmarks, (195, 1, 2))
            compile_connected(d, face_landmarks, (33, 133))
            compile_connected(d, face_landmarks, (263, 362))

            png_io = BytesIO()
            d.save_png(png_io)
            png_io.seek(0)

            png_bytes = np.asarray(bytearray(png_io.read()), dtype=np.uint8)
            conv_image = cv2.imdecode(png_bytes, cv2.IMREAD_COLOR)

            return conv_image

    return None


# test

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (800, 400))

    cv2.imshow('Capture', image)

    conv_image = process_svoji(image)

    if conv_image is not None:
        cv2.imshow('Converted', conv_image)

    if cv2.waitKey(1) == 27:
        break  # esc to quit

cv2.destroyAllWindows()
