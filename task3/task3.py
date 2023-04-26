import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
import prettytable
from segmentation import get_icon_boxes
# from hyperopt import hp, Trials, fmin, tpe, STATUS_OK, STATUS_FAIL
# import matplotlib.pyplot as plt
from typing import Tuple

SHOW = False


def draw_bounding_box(image, box, icon_num, icon_name):
    cv2.rectangle(image, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 0, 255), 1)
    cv2.putText(image, icon_num, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_4)
    cv2.putText(image, icon_name, (box[0], box[1] + box[3] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
                cv2.LINE_4)


def get_dir_files(path):
    files = [path + f for f in listdir(path) if isfile(join(path, f))]
    return files


train_image_folder = "./train_images/"
train_image_paths = get_dir_files(train_image_folder)
# TestWithoutRotations
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
    # sift = cv2.SIFT_create(nfeatures=0, nOctaveLayers=3, contrastThreshold=0.0285877, edgeThreshold=16.727215919,
    #                        sigma=1.4854635986)
    sift = cv2.SIFT_create(**parameters)
    if resize_sift_image:
        image = cv2.resize(image, (resize_sift_image, resize_sift_image), interpolation=cv2.INTER_LINEAR)
    kp, desc = sift.detectAndCompute(image, None)
    return kp, desc


def get_train_icon_features(parameters):
    train_features = []
    for path in train_image_paths:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        kp, desc = sift_get_features(image, parameters, None)
        train_image_name = path.split("/")[-1].split(".png")[0]
        train_features.append((image, train_image_name, kp, desc))
    return train_features


def get_icons(path, resize_test_scale):
    icon_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    icon_list = []
    icons = get_icon_boxes(icon_image, resize_test_scale)  # first index = image name
    for (x, y, w, h), box in icons:
        icon = icon_image[y:y + h, x: x + w]
        edited_icon = cv2.copyMakeBorder(icon, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        icon_list.append((edited_icon, (x, y, w, h)))
    return icon_list


def icon_detect_and_compute(path, parameters, resize_sift_image, resize_test_scale):
    features = []
    icons = get_icons(path, resize_test_scale)
    for icon_image, bounding_box in icons:
        kp, desc = sift_get_features(icon_image, parameters, resize_sift_image)
        features.append((kp, desc, path, bounding_box))
    return features


def calculate_iou(box_pred: Tuple[int, int, int, int], box_gt: Tuple[int, int, int, int]) -> float:
    """Calculates the intersection over union of the given prediction and ground truth boxes."""
    x1_pred, y1_pred, x2_pred, y2_pred = box_pred
    x1_gt, y1_gt, x2_gt, y2_gt = box_gt

    pred_area: int = (x2_pred - x1_pred) * (y2_pred - y1_pred)
    gt_area: int = (x2_gt - x1_gt) * (y2_gt - y1_gt)
    x1: int = max(x1_pred, x1_gt)
    y1: int = max(y1_pred, y1_gt)
    x2: int = min(x2_pred, x2_gt)
    y2: int = min(y2_pred, y2_gt)
    if x1 > x2 or y1 > y2:
        return 0.0
    intersection_area: int = (x2 - x1) * (y2 - y1)
    union_area: int = pred_area + gt_area - intersection_area
    return intersection_area / union_area if union_area > 0 else 0.0


def main(params):
    # apply SIFT to train icons
    train_kp_desc = get_train_icon_features(params['SIFT'])

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    i = 0
    accuracy = 0

    # read in test image one at a time
    for test_image in test_image_paths:
        test_predicted_icons = []
        test_image_boxes = []
        test_image_array = cv2.imread(test_image, None)
        # segment test image icons (one by one)
        for (test_icon_kp, test_icon_desc, icon_path, box) in icon_detect_and_compute(test_image, params['SIFT'],
                                                                                      params['resize_sift_image'],
                                                                                      params['resize_test_scale']):
            # get each train icon kp and desc
            for train_image, train_icon_name, train_icon_kp, train_icon_desc in train_kp_desc:

                # find any matches between the two
                matches = bf.knnMatch(test_icon_desc, train_icon_desc, k=2)

                valid_matches = []
                for match in matches:
                    if len(match) == 2:
                        m, n = match
                        if m.distance < params['distanceThreshold'] * n.distance:
                            valid_matches.append(m)

                # need at least 4 for RANSAC to work
                if len(valid_matches) < 4:
                    continue

                train_points = np.float32([train_icon_kp[m.trainIdx].pt for m in valid_matches]).reshape(-1, 1, 2)
                test_points = np.float32([test_icon_kp[m.queryIdx].pt for m in valid_matches]).reshape(-1, 1, 2)

                # return homography matrix and binary matrix of points used in homography
                homography_matrix, binary_matrix = cv2.findHomography(train_points, test_points, cv2.RANSAC,
                                                                      ransacReprojThreshold=params['RNSCThresh'])
                inliers = binary_matrix.ravel().tolist()

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

        test_predicted_icons = predicted_icons_set.intersection(actual_icons_set)
        all_icons_len = len(actual_icons_set)

        false_positive_set = predicted_icons_set.difference(actual_icons_set)
        false_negative_set = actual_icons_set.difference(predicted_icons_set)

        accuracy = round(len(test_predicted_icons) / all_icons_len * 100, 1)

        # Print out results
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
            cv2.destroyAllWindows()

        i += 1
    print(f"total accuracy: {accuracy} with PARAMS: {params}")


tuned_params = {'RNSCThresh': 10.759948009735304, 'SIFT': {'contrastThreshold': 0.004139086378011707,
                                                           'edgeThreshold': 17.310486555802317, 'nOctaveLayers': 7,
                                                           'nfeatures': 1900, 'sigma': 2.1741346835913578},
                'distanceThreshold': 0.6779285821510288, 'inliers': 4, 'resize_sift_image': 95, 'resize_test_scale': 5}

main(tuned_params)

"""
IOU Code:
name = test_image.split("/")
predicted_filename = name[1] + "-" + name[3][:-4]
with open(f"./predicted/{predicted_filename}.txt", "w") as f:
    for n in range(len(test_predicted_icons)):
        f.write(f"{test_predicted_icons[n]}, {str(test_image_boxes[n][0])}, {str(test_image_boxes[n][1])}\n")
f.close()

predicted_test_image_boxes_path = "./predicted/"
predicted_test_image_boxes = sorted(get_dir_files(predicted_test_image_boxes_path))

test_image_annotations_paths = sorted(test_image_annotations_paths)

all_all_iou = []
for i in range(len(test_image_annotations_paths)):
    print(f"Image: {test_image_annotations_paths[i]}")
    # get and sort ground truth annotations
    annotations_path = test_image_annotations_paths[i]
    test_annotations_list = get_test_annotations(annotations_path)
    test_annotations = sorted(test_annotations_list, key=lambda x: x[0])
    # print(test_annotations)

    # get and sort predicted annotations
    predicted_annotations_path = predicted_test_image_boxes[i]
    predicted_annotations_list = get_test_annotations(predicted_annotations_path)
    predicted_annotations = sorted(predicted_annotations_list, key=lambda x: x[0])
    # print(predicted_annotations)
    all_iou = []
    for n in range(len(test_annotations)):
        test_x1, test_y1 = tuple(
            int(num) for num in test_annotations[n][1].replace('(', '').replace(')', '').replace('...', '').split(', '))
        test_x2, test_y2 = tuple(
            int(num) for num in test_annotations[n][2].replace('(', '').replace(')', '').replace('...', '').split(', '))

        predicted_x1, predicted_y1 = tuple(
            int(num) for num in predicted_annotations[n][1].replace('(', '').replace(')', '').replace('...', '').split(', '))
        predicted_x2, predicted_y2 = tuple(
            int(num) for num in predicted_annotations[n][2].replace('(', '').replace(')', '').replace('...', '').split(', '))
        iou = calculate_iou((predicted_x1, predicted_y1, predicted_x2, predicted_y2), (test_x1, test_y1, test_x2, test_y2))
        all_iou.append(iou)
        all_all_iou.append(iou)

    print(f"{np.average(all_iou)}")
print(f"ALL Mean: {np.average(all_all_iou)}")
print(f"Resize & Rotate: {np.average(all_all_iou[0:21])}")
print(f"Default: {np.average(all_all_iou[21:41])}")



HyperOpt Hyperparameter Tuning Code:


param_space = {
    'RNSCThresh': hp.uniform('RNSCThresh', 10, 11),
    'distanceThreshold': hp.uniform('distanceThreshold', 0.60, 0.75),
    'resize_sift_image': hp.choice('resize_sift_image', range(5, 100, 5)),
    'inliers': hp.choice('inliers', range(3, 7)),
    'SIFT': {
        'nOctaveLayers': hp.choice('nOctaveLayers', range(5, 8)),
        'contrastThreshold': hp.uniform('contrastThreshold', 0.003, 0.005),
        'edgeThreshold': hp.uniform('edgeThreshold', 16, 17.5),
        'nfeatures': hp.choice('nfeatures', range(1400, 2000, 100)),
        'sigma': hp.uniform('sigma', 2, 2.5),
    },
    'resize_test_scale': hp.choice('resize_test_scale', range(2, 10))
}

trials = Trials()
fmin(
    fn=tune_parameters,
    space=param_space,
    algo=tpe.suggest,
    max_evals=3000,
    trials=trials
)

return {'loss': -accuracy, 'status': STATUS_OK}
"""
