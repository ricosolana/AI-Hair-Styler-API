import cv2
import mediapipe as mp
import tkinter as tk
import numpy as np
import time

root = tk.Tk()
root.geometry("800x500")
root.title("Auto Capture Application")
root.config(bg="gray")

top_frame = tk.Frame(root, width=800, height=80, bg="white")
top_frame.pack(side="top", fill="x")
label = tk.Label(top_frame, text="Auto Image Capture Application", font=('Arial', 18), bg="white")
label.pack(padx=20, pady=20)

canvas = tk.Canvas(root, width=800, height=400, bg="white")
canvas.pack()


def run_app():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(0)

    previous_pose = None
    pose_start_time = time.time()

    # Rectangular region parameters
    top_left_x, top_left_y = 250, 100
    bottom_right_x, bottom_right_y = 500, 350

    # Pose recognition and image capture flags
    recognize_pose = False
    capture_pose = False

    while cap.isOpened():
        success, image = cap.read()

        start = time.time()

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

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                nose_x = int(face_landmarks.landmark[1].x * img_w)
                nose_y = int(face_landmarks.landmark[1].y * img_h)

                # If the nose is in the center of the rectangle, initiate pose recognition and image capture
                if top_left_x < nose_x < bottom_right_x and top_left_y < nose_y < bottom_right_y:
                    recognize_pose = True
                    capture_pose = True
                else:
                    recognize_pose = False

                # Pose recognition and image capture
                if recognize_pose:
                    face_center_x = int((face_landmarks.landmark[33].x + face_landmarks.landmark[263].x) * img_w / 2)
                    face_center_y = int((face_landmarks.landmark[33].y + face_landmarks.landmark[263].y) * img_h / 2)

                    # If the face is within the rectangular region, capture images for different poses
                    if top_left_x < face_center_x < bottom_right_x and top_left_y < face_center_y < bottom_right_y:
                        if capture_pose:
                            for idx, lm in enumerate(face_landmarks.landmark):
                                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                                    face_2d.append([x, y])
                                    face_3d.append([x, y, lm.z])

                            face_2d = np.array(face_2d, dtype=np.float64)
                            face_3d = np.array(face_3d, dtype=np.float64)

                            focal_length = 1 * img_w
                            cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                                   [0, focal_length, img_w / 2],
                                                   [0, 0, 1]])

                            dist_matrix = np.zeros((4, 1), dtype=np.float64)

                            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                            rmat, jac = cv2.Rodrigues(rot_vec)

                            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

                            x_angle = angles[0] * 360
                            y_angle = angles[1] * 360
                            z_angle = angles[2] * 360

                            if y_angle < -10:
                                current_pose = "Looking Left"
                            elif y_angle > 10:
                                current_pose = "Looking Right"
                            else:
                                current_pose = "Forward"

                            if current_pose != previous_pose:
                                pose_start_time = time.time()
                            elif time.time() - pose_start_time >= 0.5:
                                # Capture and save only the region inside the rectangular boundary
                                face_crop = image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
                                cv2.imwrite(f'{current_pose}_face.jpg', cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))

                                # Reset the timer and pose_start_time
                                pose_start_time = time.time()

                            previous_pose = current_pose
                            capture_pose = False

        # Display code
        end = time.time()
        totalTime = end - start

        fps = 1 / totalTime
        print("FPS: ", fps)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.putText(image, f'FPS: {int(fps)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

        cv2.imshow('Head Pose Estimation', image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break


button = tk.Button(top_frame, text="Click To Run the Application", font=('Arial', 12), command=run_app, bg="gray")
button.pack(padx=5, pady=5)

root.mainloop()
