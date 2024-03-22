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
    start_time = time.time()

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (800, 400))
    image1 = image
    image1.flags.writeable = False

    results = face_mesh.process(image1)

    image1.flags.writeable = True

    img_h, img_w, img_c = image1.shape
    face_3d = []
    face_2d = []

    # Draw a rectangle on the image with a dot in the center
    cv2.rectangle(image1, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (255, 0, 0), 2)
    center_x = int((top_left_x + bottom_right_x) / 2)
    center_y = int((top_left_y + bottom_right_y) / 2)
    # cv2.circle(image1, (center_x, center_y), 5, (0, 255, 0), -1)

    #recognize_pose = False

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            nose_x = int(face_landmarks.landmark[1].x * img_w)
            nose_y = int(face_landmarks.landmark[1].y * img_h)

            # If the nose is in the center of the rectangle, initiate pose recognition and image capture
            if top_left_x < nose_x < bottom_right_x and top_left_y < nose_y < bottom_right_y:
                recognize_pose = True
            else:
                recognize_pose = False

            # Pose recognition and image capture
            if recognize_pose:
                face_center_x = int((face_landmarks.landmark[33].x + face_landmarks.landmark[263].x) * img_w / 2)
                face_center_y = int((face_landmarks.landmark[33].y + face_landmarks.landmark[263].y) * img_h / 2)

                # If the face is within the rectangular region, capture images for different poses
                if top_left_x < face_center_x < bottom_right_x and top_left_y < face_center_y < bottom_right_y:
                    # instead of decomposing points to 3d, draw 2d?
                    # this will be rawly-rastered; the perspective will be baked, pose-as-is
                    # maybe later add a rotation / normalizing feature



                    #d.save_svg(fname='tmp1.svg')

                    # nose
                    #   195(4, 5? top) none exact
                    #   1(tip)
                    #   2(bottom)

                    nose_top = face_landmarks.landmark[195]
                    nose_mid = face_landmarks.landmark[1]
                    nose_bot = face_landmarks.landmark[2]

                    right_eye1 = face_landmarks.landmark[33]
                    right_eye2 = face_landmarks.landmark[133]

                    left_eye1 = face_landmarks.landmark[263]
                    left_eye2 = face_landmarks.landmark[362]

                    d = drawsvg.Drawing(800, 400) #, origin='center')

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

                    return {

                    }, conv_image

                    #d.save_svg(fname='tmp2.svg')



                    """
                    if success:
                        return {
                            'pose': current_pose,
                            'x_angle': x_angle,
                            'y_angle': y_angle,
                            'z_angle': z_angle,
                            #'frame': enc,
                            'time': processing_time
                        }, enc.tobytes()
                    else:
                        return {'message': 'Image encoding failure'}
                    """
    else:
        return {'message': 'No detectable landmarks'}

    return {'message': 'Landmark is not utilized'}


# test

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()

    #image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (800, 400))

    cv2.imshow('Capture', image)

    data = process_svoji(image)

    if data[1] is not None:
        cv2.imshow('Converted', data[1])

    if cv2.waitKey(1) == 27:
        break  # esc to quit

cv2.destroyAllWindows()
