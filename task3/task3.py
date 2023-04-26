import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
import prettytable
from segmentation import get_icon_boxes

SHOW = False


def draw_bounding_box(image, box, icon_num, icon_name):
    cv2.rectangle(image, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 0, 255), 1)
    cv2.putText(image, icon_num, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_4)
    cv2.putText(image, icon_name, (box[0], box[1] + box[3] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
                cv2.LINE_4)


def get_dir_files(path):
    files = [path + f for f in listdir(path) if isfile(join(path, f))]
    return files


# Get the training image paths
train_image_folder = "./train_images/"
train_image_paths = get_dir_files(train_image_folder)

# get both test dataset paths
default_test_image_folder = "./TestWithoutRotations/images/"
additional_test_image_folder = "./Task3AdditionalTestDataset/images/"
test_image_paths = get_dir_files(default_test_image_folder) + get_dir_files(additional_test_image_folder)
default_test_image_annotations = "./TestWithoutRotations/annotations/"
additional_test_image_annotations = "./Task3AdditionalTestDataset/annotations/"
test_image_annotations_paths = get_dir_files(default_test_image_annotations) + get_dir_files(
    additional_test_image_annotations)


# gets icons and locations from txt file
def get_test_annotations(path):
    test_annotations = []
    with open(path, "r") as f:
        for line in f:
            split_line = line.split(", ")
            test_annotations.append(
                [split_line[0], split_line[1] + ", " + split_line[2], split_line[3] + ", " + split_line[4].strip("\n")])
    return test_annotations


def sift_get_features(image, parameters, resize_sift_image):
    sift = cv2.SIFT_create(**parameters)
    if resize_sift_image:
        image = cv2.resize(image, (resize_sift_image, resize_sift_image), interpolation=cv2.INTER_LINEAR)
    kp, desc = sift.detectAndCompute(image, None)
    return kp, desc


# iterate through training images and get keypoints and descriptors
def get_train_icon_features(parameters):
    train_features = []
    for path in train_image_paths:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        kp, desc = sift_get_features(image, parameters, None)
        train_image_name = path.split("/")[-1].split(".png")[0]
        train_features.append((image, train_image_name, kp, desc))
    return train_features


# get test image icons using segmentation
def get_icons(path, resize_segment_icon):
    icon_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    icon_list = []
    icons = get_icon_boxes(icon_image, resize_segment_icon)  # first index = image name
    for (x, y, w, h), box in icons:
        icon = icon_image[y:y + h, x: x + w]
        edited_icon = cv2.copyMakeBorder(icon, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        icon_list.append((edited_icon, (x, y, w, h)))
    return icon_list


# iterate through testing images and get keypoints and descriptors
def icon_detect_and_compute(path, parameters, resize_sift_image, resize_segment_icon):
    features = []
    icons = get_icons(path, resize_segment_icon)
    for icon_image, bounding_box in icons:
        kp, desc = sift_get_features(icon_image, parameters, resize_sift_image)
        features.append((kp, desc, path, bounding_box))
    return features


def main(params):
    # apply SIFT to train icons
    train_kp_desc = get_train_icon_features(params['SIFT'])

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    i = 0
    accuracy = 0
    precision = []
    recall = []
    # read in test image one at a time
    for test_image in test_image_paths:
        test_predicted_icons = []
        test_image_boxes = []
        test_image_array = cv2.imread(test_image, None)
        # segment test image icons (one by one)
        for (test_icon_kp, test_icon_desc, icon_path, box) in icon_detect_and_compute(test_image, params['SIFT'],
                                                                                      params['resize_sift_image'],
                                                                                      params['resize_segment_icon']):
            # get each train icon kp and desc
            for train_image, train_icon_name, train_icon_kp, train_icon_desc in train_kp_desc:

                # find any matches between the two
                matches = bf.knnMatch(test_icon_desc, train_icon_desc, k=2)

                valid_matches = []
                for match in matches:
                    if len(match) == 2:
                        m, n = match
                        if m.distance < params['distanceThreshold'] * n.distance:
                            valid_matches.append([m])

                # need at least 4 for RANSAC to work
                if len(valid_matches) < 4:
                    continue

                train_points = np.float32([train_icon_kp[m[0].trainIdx].pt for m in valid_matches]).reshape(-1, 1, 2)
                test_points = np.float32([test_icon_kp[m[0].queryIdx].pt for m in valid_matches]).reshape(-1, 1, 2)

                # return homography matrix and binary matrix of points used in homography
                homography_matrix, binary_matrix = cv2.findHomography(train_points, test_points, cv2.RANSAC,
                                                                      ransacReprojThreshold=params['RNSCThresh'])
                inliers = binary_matrix.ravel().tolist()

                # check if accurate enough match
                if sum(inliers) > params['inliers']:
                    # split train icon name into number and name
                    icon_num = train_icon_name.split("-")[0]
                    icon_name = "-".join(train_icon_name.split("-")[1:])

                    draw_bounding_box(test_image_array, box, icon_num, icon_name)
                    test_predicted_icons.append(icon_name)
                    test_image_boxes.append(((box[0], box[1]), (box[0] + box[2], box[1] + box[3])))

        # Compare predicted to actual icons
        predicted_icons_set = set(test_predicted_icons)
        test_annotated_features = get_test_annotations(test_image_annotations_paths[i])
        actual_icons_set = set([f[0] for f in test_annotated_features])

        true_positive_icons = predicted_icons_set.intersection(actual_icons_set)
        all_icons_len = len(actual_icons_set)

        false_positive_set = predicted_icons_set.difference(actual_icons_set)
        false_negative_set = actual_icons_set.difference(predicted_icons_set)

        accuracy = round(len(true_positive_icons) / all_icons_len * 100, 1)

        # Print out results
        test_image_name = test_image.split("/")[-1][:-4]
        x = prettytable.PrettyTable(hrules=1)
        x.field_names = ["File: " + test_image_name, "Accuracy: " + str(accuracy) + "%"]
        x.add_row(["True Positives: " + ", ".join(true_positive_icons), "False Positives: "
                   + (", ".join(false_positive_set) if len(false_positive_set) != 0 else "N/A")])
        x.add_row(["False Negatives: " +
                   (", ".join(false_negative_set) if len(false_negative_set) != 0 else "N/A"), "True Negatives: N/A"])
        print(x)

        precision.append(len(true_positive_icons) / (len(true_positive_icons) + len(false_positive_set)))
        recall.append(len(true_positive_icons) / (len(true_positive_icons) + len(false_negative_set)))

        if SHOW:
            cv2.imshow('image', test_image_array)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        i += 1


tuned_params = {'RNSCThresh': 10.759948009735304, 'SIFT': {'contrastThreshold': 0.004139086378011707,
                                                           'edgeThreshold': 17.310486555802317, 'nOctaveLayers': 7,
                                                           'nfeatures': 1900, 'sigma': 2.1741346835913578},
                'distanceThreshold': 0.6779285821510288, 'inliers': 4, 'resize_sift_image': 95,
                'resize_segment_icon': 5}

main(tuned_params)
