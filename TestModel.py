import cv2 as cv
from vision import Vision
from matplotlib import pyplot as plt

# load the trained model
cascade_limestone = cv.CascadeClassifier('2.3cascade/cascade.xml')

# load an empty Vision class
vision_limestone = Vision(None)

# Load the image
screenshot = cv.imread('testing/345_L.png')

# do object detection
rectangles = cascade_limestone.detectMultiScale(screenshot)

# draw the detection results onto the original image
detection_image = vision_limestone.draw_rectangles(screenshot, rectangles)

# show number of object detected
font = cv.FONT_HERSHEY_SIMPLEX
cv.putText(detection_image, "Detected: " + str(len(rectangles)), (100, 100), font, 2, (0, 0, 255), 5, cv.LINE_AA)

# display the images
plt.imshow(detection_image)
plt.show()