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


import cairosvg


from io import BytesIO
import cv2
import mediapipe as mp
import numpy as np
import time
import drawsvg

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Rectangular region parameters
top_left_x, top_left_y = 250, 100
bottom_right_x, bottom_right_y = 500, 350


def process_svoji(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(image)

    img_h, img_w, img_c = image.shape

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            nose_top = face_landmarks.landmark[195]
            nose_mid = face_landmarks.landmark[1]
            nose_bot = face_landmarks.landmark[2]

            right_eye1 = face_landmarks.landmark[33]
            right_eye2 = face_landmarks.landmark[133]

            left_eye1 = face_landmarks.landmark[263]
            left_eye2 = face_landmarks.landmark[362]

            d = drawsvg.Drawing(800, 400)

            # Nose
            p = drawsvg.Path(stroke_width=2, stroke='lime', fill='black', fill_opacity=0.2)
            p.M(nose_top.x * img_w, nose_top.y * img_h)  # Start path at point (nose_top)
            #p.C(30, 10, 30, -50, 70, -20)  # Draw a curve to (70, -20)
            p.L(nose_mid.x * img_w, nose_mid.y * img_h)
            p.L(nose_bot.x * img_w, nose_bot.y * img_h)

            # Right eye
            p.M(right_eye1.x * img_w, right_eye1.y * img_h)
            p.L(right_eye2.x * img_w, right_eye2.y * img_h)

            # Left eye
            p.M(left_eye1.x * img_w, left_eye1.y * img_h)
            p.L(left_eye2.x * img_w, left_eye2.y * img_h)

            d.append(p)

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
