# Install GTK https://github.com/Kozea/CairoSVG/issues/371#issuecomment-1539570439
# If that fails, install cairo using vcpkg; ensure working directory contains no spaces

import math
from io import BytesIO

import cairosvg
import cv2
import drawsvg
import mediapipe as mp
import numpy as np
import pyvirtualcam

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


def compile_linear(d: drawsvg.Drawing, face_landmarks, indices,
                   smooth=False, fill='black', stroke_width=2, stroke='none'):
    p = drawsvg.Path(stroke_width=stroke_width, stroke=stroke, fill=fill)

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


"""
@ output_type = 'png' | 'jpg' | 'np' | 'svg'
TODO input_format=None, expected input
"""
def process_svoji(image, output_format='np'):
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #cv2.COLORYUV4202RGB
    #image = cv2.cvtColor(image, )
    #image = cv2.cvtColor()

    results = face_mesh.process(image)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            height, width, _ = image.shape
            d = drawsvg.Drawing(width, height)

            right_eye1 = face_landmarks.landmark[159]
            right_eye2 = face_landmarks.landmark[153]
            left_eye1 = face_landmarks.landmark[386]
            left_eye2 = face_landmarks.landmark[380]

            # get distance between each per-eye point
            # x^2 + y^2 = z^2
            #right_radius = math.sqrt(((right_eye2.x - right_eye1.x)**2) + ((right_eye2.y - right_eye1.y)**2)) * d.width
            #left_radius = math.sqrt(((left_eye2.x - left_eye1.x) ** 2) + ((left_eye2.y - left_eye1.y) ** 2)) * d.width

            # using center of face, get far point on side of face for radius
            # make this the circle "face"
            face_center = face_landmarks.landmark[1]
            face_outer = face_landmarks.landmark[10]
            face_radius = math.sqrt((face_center.x-face_outer.x)**2 + (face_center.y-face_outer.y)**2) * d.width  # wrong scale
            d.append(drawsvg.Circle(face_center.x * d.width, face_center.y * d.height, face_radius,
                                    fill='orange'))

            right_radius = 4
            left_radius = 4

            right_eye_point = ((right_eye1.x + right_eye2.x) / 2, (right_eye1.y + right_eye2.y) / 2)
            left_eye_point = ((left_eye1.x + left_eye2.x) / 2, (left_eye1.y + left_eye2.y) / 2)

            # TODO scaling of radius with width is not correct
            d.append(drawsvg.Circle(right_eye_point[0] * d.width, right_eye_point[1] * d.height, right_radius,
                                 fill='white'))
            d.append(drawsvg.Circle(left_eye_point[0] * d.width, left_eye_point[1] * d.height, left_radius,
                                    fill='white'))

            #compile_linear(d, face_landmarks, (57, 178, 402, 287), smooth=True)

            compile_linear(d, face_landmarks, (78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308),
                           smooth=True, fill='white')

            #compile_linear(d, face_landmarks, (57, 91, 321, 287), smooth=False)

            # get face outline coordinates
            #face_landmarks

            # 57, 287

            # 178, 402

            #compile_linear(d, face_landmarks, (195, 1, 2), smooth=True)  # nose
            #compile_linear(d, face_landmarks, (263, 362))
            #compile_linear(d, face_landmarks, (33, 133))

            bytes_io = BytesIO()
            if output_format == 'np':
                d.save_png(bytes_io)
                bytes_io.seek(0)
                png_bytes = np.asarray(bytearray(bytes_io.read()), dtype=np.uint8)

                return cv2.imdecode(png_bytes, cv2.IMREAD_COLOR)
            elif output_format == 'png':
                d.save_png(bytes_io)
                bytes_io.seek(0)
                return bytearray(bytes_io.read())
            elif output_format in ('jpg', 'jpeg'):
                d.save_png(bytes_io)
                bytes_io.seek(0)
                png_bytes = np.asarray(bytearray(bytes_io.read()), dtype=np.uint8)

                conv_image = cv2.imdecode(png_bytes, cv2.IMREAD_COLOR)

                res, png_bytes = cv2.imencode('.jpg', conv_image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                return bytearray(png_bytes)
            elif output_format == 'svg':
                return d.as_svg().encode('utf-8')
                #d.save_svg(bytes_io)
                #bytes_io.seek(0)
                #return bytearray(bytes_io.read())

            return None

    return None


if __name__ == '__main__':
    UPSCALE_RATIO = (800, 400)

    cap = cv2.VideoCapture(0)

    if cap.isOpened():
        cam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cam = pyvirtualcam.Camera(width=cam_width,
                                  height=cam_height,
                                  fps=int(cap.get(cv2.CAP_PROP_FPS)))

        while cap.isOpened():
            success, image = cap.read()

            image = cv2.flip(image, 1)

            conv_image = process_svoji(image)

            scaled_image = cv2.resize(image, UPSCALE_RATIO)
            cv2.imshow('Capture', scaled_image)

            if conv_image is not None:
                scaled_conv_image = cv2.resize(conv_image, UPSCALE_RATIO)
                cv2.imshow('Converted', scaled_conv_image)

                cam.send(conv_image)

            if cv2.waitKey(1) == 27:
                break  # esc to quit

    cv2.destroyAllWindows()
