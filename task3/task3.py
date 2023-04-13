# TODO: fix change training icon names - DONE
# TODO: screen shot missing trash - DONE
# TODO: some of the icon texts go off screen - do a check to see if the pixel vals will be over the 512x512
#  - task 2 dataset doesn't worry about this
# TODO: find match distance parameter to solve FPs TODO: find SIFT parameters to fix False Positives
# TODO: manage rotations
# TODO: run through images in proper order

import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
from segmentation import get_icon_boxes
import prettytable
SHOW = True


def draw_bounding_box(image, box, icon_num, icon_name):
    cv2.rectangle(image, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 0, 255), 1)
    cv2.putText(image, icon_num, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_4)
    cv2.putText(image, icon_name, (box[0], box[1] + box[3] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_4)


def get_dir_files(path):
    files = [path + f for f in listdir(path) if isfile(join(path, f))]
    return files


train_image_folder = "./train_images/"
train_image_paths = get_dir_files(train_image_folder)
test_image_folder = "./TestWithoutRotations/images/"
test_image_paths = get_dir_files(test_image_folder)
test_image_annotations = "./TestWithoutRotations/annotations/"
test_image_annotations_paths = get_dir_files(test_image_annotations)


# gets icons and locations from txt file
def get_test_annotations(path):
    test_annotations = []
    with open(path, "r") as f:
        for line in f:
            split_line = line.split(", ")
            test_annotations.append(
                [split_line[0], split_line[1] + ", " + split_line[2], split_line[3] + ", " + split_line[4].strip("\n")])
    return test_annotations


def sift_get_features(image):
    sift = cv2.SIFT_create()  # TODO: find best parameters
    kp, desc = sift.detectAndCompute(image, None)
    return kp, desc


def get_train_icon_features():
    train_features = []
    for path in train_image_paths:
        image = cv2.imread(path, None)
        kp, desc = sift_get_features(image)
        train_image_name = path.split("/")[-1].split(".png")[0]
        train_features.append((train_image_name, kp, desc))
    return train_features


def get_icons(path):
    icon_image = cv2.imread(path, None)
    icon_list = []
    icons = get_icon_boxes(icon_image)  # first index = image name
    for x, y, w, h in icons:
        icon = icon_image[y:y + h, x: x + w]
        edited_icon = cv2.copyMakeBorder(icon, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        icon_list.append((edited_icon, (x, y, w, h)))
    return icon_list


def icon_detect_and_compute(path):
    features = []
    icons = get_icons(path)
    for icon_image, bounding_box in icons:
        kp, desc = sift_get_features(icon_image)
        features.append((kp, desc, path, bounding_box))

    return features


# apply SIFT to train icons
train_kp_desc = get_train_icon_features()

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
FPs = []
FNs = []
i = 0

# read in test image one at a time
for test_image in test_image_paths:
    test_predicted_icons = []
    test_image_array = cv2.imread(test_image)
    # segment test image icons (one by one)
    for (test_icon_kp, test_icon_desc, icon_path, box) in icon_detect_and_compute(test_image):
        # get each train icon kp and desc
        for train_icon_name, train_icon_kp, train_icon_desc in train_kp_desc:

            # find any matches between the two
            matches = bf.knnMatch(test_icon_desc, train_icon_desc, k=2)

            valid_matches = []
            for match in matches:
                if len(match) == 2:
                    m, n = match
                    if m.distance < 0.8 * n.distance:  # TODO: tune 0.8 val
                        valid_matches.append(m)

            # need at least 4 for RANSAC to work
            if len(valid_matches) < 4:
                continue

            train_points = np.float32([train_icon_kp[m.trainIdx].pt for m in valid_matches]).reshape(-1, 1, 2)
            test_points = np.float32([test_icon_kp[m.queryIdx].pt for m in valid_matches]).reshape(-1, 1, 2)

            # return homography matrix and binary matrix of points used in homography
            homography_matrix, binary_matrix = cv2.findHomography(train_points, test_points, cv2.RANSAC)
            inliers = binary_matrix.ravel().tolist()

            if sum(inliers) > 5:
                # split train icon name into number and name
                icon_num = train_icon_name.split("-")[0]
                icon_name = "-".join(train_icon_name.split("-")[1:])

                draw_bounding_box(test_image_array, box, icon_num, icon_name)
                test_predicted_icons.append(icon_name)

    predicted_icons_set = set(test_predicted_icons)
    test_annotated_features = get_test_annotations(test_image_annotations_paths[i])
    actual_icons_set = set([f[0] for f in test_annotated_features])

    test_predicted_icons = predicted_icons_set.intersection(actual_icons_set)
    all_icons_len = len(actual_icons_set)

    false_positive_set = predicted_icons_set.difference(actual_icons_set)
    false_negative_set = actual_icons_set.difference(predicted_icons_set)

    accuracy = round(len(test_predicted_icons) / all_icons_len * 100, 1)
    test_image_name = test_image.split("/")[-1][:-4]

    x = prettytable.PrettyTable(hrules=1)
    x.field_names = ["File: " + test_image_name, "Accuracy: " + str(accuracy) + "%"]
    x.add_row(["True Positives: " + ", ".join(test_predicted_icons), "False Positives: "
               + (", ".join(false_positive_set) if len(false_positive_set) != 0 else "N/A")])
    x.add_row(["False Negatives: " +
               (", ".join(false_negative_set) if len(false_negative_set) != 0 else "N/A"), "True Negatives: N/A"])
    print(x)

    if SHOW:
        cv2.imshow('image', test_image_array)
        cv2.waitKey(0)

    i += 1

"""
TODO:
 - get test image features
    - some preprocessing of icons
    - take each icon using bounding box coordinates
    - apply SIFT detect and compute
 - match segments
    - bruteforcer with knnMatch
    - filter any poor matches
    - apply RANSAC
        - homography matrix
 - manage rotation
 - manage resizing?
 - tune match distance value
 - tune SIFT parameters
"""
