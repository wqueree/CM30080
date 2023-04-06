import cv2
from task3gen import sift

image = cv2.imread('001-lighthouse.png', 0)
image = sift.computeKeypointsAndDescriptors(image)
sift.display_image(image)