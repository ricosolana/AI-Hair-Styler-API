# import the necessary packages
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2

# im using cv4
#stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", type=str, required=True,
	help="path to input directory of images to stitch")
#ap.add_argument("-o", "--output", type=str, required=True,
	#help="path to the output image")
args = vars(ap.parse_args())

# grab the paths to the input images and initialize our images list
print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images(args["images"])))
images = []
# loop over the image paths, load each one, and add them to our
# images to stitch list
for imagePath in imagePaths:
	image = cv2.imread(imagePath)
	images.append(image)

# initialize OpenCV's image stitcher object and then perform the image
# stitching
print("[INFO] stitching images...")
stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
(status, stitched) = stitcher.stitch(images)

# if the status is '0', then OpenCV successfully performed image
# stitching
if status == 0:
	# write the output stitched image to disk
	#cv2.imwrite(args["output"], stitched)

	height, width, _ = stitched.shape

	cv2.fisheye.calibrate()

	nk = k.copy()
	nk[0, 0] = k[0, 0] / 2
	nk[1, 1] = k[1, 1] / 2
	# Just by scaling the matrix coefficients!

	map1, map2 = cv2.fisheye.initUndistortRectifyMap(k, d, np.eye(3), nk, (800, 600),
													 cv2.CV_16SC2)  # Pass k in 1st parameter, nk in 4th parameter
	nemImg = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

	equirectangular = np.zeros((height, width, 3), dtype=np.uint8)

	# Calculate the equirectangular projection
	for y in range(height):
		for x in range(width):
			lon = 2 * np.pi * x / width - np.pi
			lat = np.pi * y / height - np.pi / 2
			x_equi = int(width * (lon + np.pi) / (2 * np.pi))
			y_equi = int(height * (np.pi / 2 - lat) / np.pi)
			if x_equi < width and y_equi < height:
				equirectangular[y_equi, x_equi] = stitched[y, x]

	# Step 2: Apply fisheye correction
	# Define the fisheye correction parameters
	k1 = -0.2  # Radial distortion coefficient
	k2 = 0.0  # Tangential distortion coefficient
	p1 = 0.0  # Tangential distortion coefficient
	p2 = 0.0  # Tangential distortion coefficient

	# Create the distortion coefficients matrix
	distCoeffs = np.array([k1, k2, p1, p2], dtype=np.float64)

	# Create the camera matrix
	cameraMatrix = np.array([[width / (2 * np.pi), 0, width / 2],
							 [0, height / np.pi, height / 2],
							 [0, 0, 1]], dtype=np.float64)

	# Create the distortion coefficients matrix
	distCoeffs = np.array([k1, k2, p1, p2], dtype=np.float64)

	# Create the map for undistortion
	map1, map2 = cv2.initUndistortRectifyMap(cameraMatrix, distCoeffs, None, cameraMatrix, (width, height),
											 cv2.CV_32FC1)

	# Apply the fisheye correction
	corrected = cv2.remap(equirectangular, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
	####resized_image = cv2.resize(equirectangular, (0, 0), fx=0.5, fy=0.5)

	cv2.namedWindow('Stitched', cv2.WINDOW_NORMAL)

	# might need to go after
	cv2.imshow("Stitched", corrected)

	(x1, y1, w1, h1) = cv2.getWindowImageRect('Stitched')
	cv2.resizeWindow('Stitched', w1//4, h1//4)

	#cv2.imshow("Stitched", stitched)
	cv2.waitKey(0)
# otherwise the stitching failed, likely due to not enough keypoints)
# being detected
else:
	print("[INFO] image stitching failed ({})".format(status))
