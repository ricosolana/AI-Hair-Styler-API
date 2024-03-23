# Install GTK https://github.com/Kozea/CairoSVG/issues/371#issuecomment-1539570439
# If that fails, install cairo using vcpkg; ensure working directory contains no spaces

import math
from io import BytesIO

import cv2
import drawsvg
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)


# https://gist.github.com/aconz2/34017193c89c8b285fef837bb2b87cd9
conversion_matrix = np.array([
    [0,       6,       0,       0],
    [-1,      6,       1,       0],
    [0,       1,       6,      -1],
    [0,       0,       6,       0],
    ])


def catmull_rom_to_bezier_vectorized(points):
    """
    Same as above but uses numpy.
    points: np.array of shape (N, 2)
    returns: np.array of shape (N - 1, 4, 2)
    """
    if len(points) == 2:
        return [(points[0], points[0], points[1], points[1])]
    A = np.empty((len(points) - 1, 4, 2))
    for i in range(1, len(points) - 2):
        A[i, :] = (points[i - 1], points[i], points[i + 1], points[i + 2])
    A[0, :] = (points[0], points[0], points[1], points[2])
    A[-1, :] = (points[-3], points[-2], points[-1], points[-1])

    return np.matmul(conversion_matrix, A) / 6


def compile_linear(d: drawsvg.Drawing, face_landmarks, indices, smooth=False):
    p = drawsvg.Path(stroke_width=2, stroke='lime', fill='black', fill_opacity=0.2)

    # first point
    p.M(face_landmarks.landmark[indices[0]].x * d.width, face_landmarks.landmark[indices[0]].y * d.height)

    if smooth:
        # Extract x and y coordinates using advanced indexing
        x_coords = np.array([face_landmarks.landmark[i].x for i in indices])
        y_coords = np.array([face_landmarks.landmark[i].y for i in indices])

        # Combine x and y coordinates into a single 2D array
        points = np.column_stack((x_coords, y_coords))
        """
        points = []
        for i in indices:
            points.append((face_landmarks.landmark[i].x, face_landmarks.landmark[i].y))
        """
        cubic_points = catmull_rom_to_bezier_vectorized(points)

        # catmull returns 1 fewer than input points
        # first and last element per curve correspond to previous and next point

        for point in cubic_points[:-1]:
            _, c1, c2, b = point
            p.C(c1[0] * d.width, c1[1] * d.height,
                c2[0] * d.width, c2[1] * d.height,
                b[0] * d.width, b[1] * d.height)
    else:
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

            right_eye1 = face_landmarks.landmark[159]
            right_eye2 = face_landmarks.landmark[153]
            left_eye1 = face_landmarks.landmark[386]
            left_eye2 = face_landmarks.landmark[380]

            # get distance between each per-eye point
            # x^2 + y^2 = z^2
            right_radius = math.sqrt(((right_eye2.x - right_eye1.x)**2) + ((right_eye2.y - right_eye1.y)**2))
            left_radius = math.sqrt(((left_eye2.x - left_eye1.x) ** 2) + ((left_eye2.y - left_eye1.y) ** 2))

            right_eye_point = ((right_eye1.x + right_eye2.x) / 2, (right_eye1.y + right_eye2.y) / 2)
            left_eye_point = ((left_eye1.x + left_eye2.x) / 2, (left_eye1.y + left_eye2.y) / 2)

            # TODO scaling of radius with width is not correct
            d.append(drawsvg.Circle(right_eye_point[0] * d.width, right_eye_point[1] * d.height, right_radius * d.width,
                                 fill='lime', stroke_width=1, stroke='black'))
            d.append(drawsvg.Circle(left_eye_point[0] * d.width, left_eye_point[1] * d.height, left_radius * d.width,
                                    fill='lime', stroke_width=1, stroke='black'))

            compile_linear(d, face_landmarks, (57, 178, 402, 287), smooth=True)

            # 57, 287

            # 178, 402




            """
            compile_connected(d, face_landmarks, (195, 1, 2))
            compile_connected(d, face_landmarks, (33, 133))
            compile_connected(d, face_landmarks, (263, 362))
            """

            png_io = BytesIO()
            d.save_png(png_io)
            png_io.seek(0)

            png_bytes = np.asarray(bytearray(png_io.read()), dtype=np.uint8)
            conv_image = cv2.imdecode(png_bytes, cv2.IMREAD_COLOR)

            return conv_image

    return None


if __name__ == '__main__':
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
