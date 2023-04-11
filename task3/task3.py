import cv2
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from segmentation import get_icon_boxes


def get_dir_images(path):
    onlyfiles = [path + f for f in listdir(path) if isfile(join(path, f))]
    return onlyfiles


train_image_folder = "./train_images/"
train_image_paths = get_dir_images(train_image_folder)
test_image_folder = "./TestWithoutRotations/images/"
test_image_paths = get_dir_images(test_image_folder)


def image_detect_and_compute(image, resize_val):
    image = cv2.imread(image, None)
    sift = cv2.SIFT_create()
    if resize_val > 0:
        image = cv2.resize(image, (resize_val, resize_val), interpolation=cv2.INTER_LINEAR)
    kp, desc = sift.detectAndCompute(image, None)
    return kp, desc


train_kp_desc = []
for image in train_image_paths:
    kp, desc = image_detect_and_compute(image, 0)
    train_kp_desc.append((image.strip(".png").strip("/train_images/"), kp, desc))

# print(train_kp_desc)

icon_coords = []
for image in test_image_paths:
    icon_coords.append(get_icon_boxes(image))

print(icon_coords)

"""
TODO:
 - get test image features
    - take each icon using bounding box coordinates
    - apply SIFT detect and compute
 - match segments
    - bruteforcer with knnMatch
    - filter any poor matches
    - apply RANSAC
        - homography matrix
        - inliers
"""
