import cv2
import numpy as np
import argparse
from imutils import paths
from defisheye import defisheye

# Load images
#img1 = cv2.imread("image1.png")
#img2 = cv2.imread("image2.png")
#img3 = cv2.imread("image3.png")

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

# Stitch images
stitcher = cv2.Stitcher.create()
status, result = stitcher.stitch(images)

# Check stitching status
if status == cv2.Stitcher_OK:
    # Inverse project to equirectangular projection
    width = result.shape[1]
    height = width // 2

    print(width) #used print statement to ensure  these exist
    print(height) #used print statement to ensure  these exist
    print(type(result[1]))# just to check it



    map_x, map_y = cv2.convertMaps(
        np.zeros((height, width), np.float32),
        np.float32(result[1]),
        cv2.CV_32FC1
    )  # had to cast result[1] to np.float32
    #note:
    #convertMaps() see https://www.opencv.org.cn/opencvdoc/2.3.2/html/modules/imgproc/doc/geometric_transformations.html
    #see comment at the bottom also.
    inverse_panorama = cv2.remap(result[0], map_x, map_y, cv2.INTER_CUBIC)

    #cv2.namedWindow('Inverse Panorama', cv2.WINDOW_NORMAL)
    # Display the inverse panorama
    cv2.imshow("Inverse Panorama", inverse_panorama)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Stitching failed. Unable to create inverse panorama.")
