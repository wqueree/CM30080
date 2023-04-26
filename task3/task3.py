import cv2
import numpy as np
from os import listdir
from argparse import ArgumentParser, Namespace
from os.path import isfile, join
import prettytable
from typing import Tuple
from segmentation import get_icon_boxes

SHOW = False
EVAL = True


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


def draw_bounding_box(image, box, icon_num, icon_name):
    cv2.rectangle(image, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 0, 255), 1)
    cv2.putText(
        image,
        icon_num,
        (box[0], box[1] - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        1,
        cv2.LINE_4,
    )
    cv2.putText(
        image,
        icon_name,
        (box[0], box[1] + box[3] + 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        1,
        cv2.LINE_4,
    )


def get_dir_files(path):
    files = [path + f for f in listdir(path) if isfile(join(path, f))]
    return files


parser: ArgumentParser = ArgumentParser()

parser.add_argument("--train_image_directory", type=str, default="./train_images/")
parser.add_argument("--test_image_directory", type=str, default="./TestWithoutRotations/images/")
parser.add_argument(
    "--test_annotation_directory", type=str, default="./TestWithoutRotations/annotations/"
)
parser.add_argument(
    "--additional_test_image_directory", type=str, default="./Task3AdditionalTestDataset/images/"
)
parser.add_argument(
    "--additional_test_annotation_directory",
    type=str,
    default="./Task3AdditionalTestDataset/annotations/",
)

args: Namespace = parser.parse_args()

# Get the training image paths
train_image_folder = args.train_image_directory
train_image_paths = get_dir_files(train_image_folder)

# get both test dataset paths
default_test_image_folder = args.test_image_directory
additional_test_image_folder = args.additional_test_image_directory
test_image_paths = get_dir_files(default_test_image_folder) + get_dir_files(
    additional_test_image_folder
)
default_test_image_annotations = args.test_annotation_directory
additional_test_image_annotations = args.additional_test_annotation_directory
test_image_annotations_paths = get_dir_files(default_test_image_annotations) + get_dir_files(
    additional_test_image_annotations
)


# gets icons and locations from txt file
def get_test_annotations(path):
    test_annotations = []
    with open(path, "r") as f:
        for line in f:
            split_line = line.split(", ")
            test_annotations.append(
                [
                    split_line[0],
                    split_line[1] + ", " + split_line[2],
                    split_line[3] + ", " + split_line[4].strip("\n"),
                ]
            )
    return test_annotations


def sift_get_features(image, parameters, resize_sift_image):
    sift = cv2.SIFT_create(**parameters)
    if resize_sift_image:
        image = cv2.resize(
            image, (resize_sift_image, resize_sift_image), interpolation=cv2.INTER_LINEAR
        )
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
        icon = icon_image[y : y + h, x : x + w]
        edited_icon = cv2.copyMakeBorder(
            icon, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )
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
    train_kp_desc = get_train_icon_features(params["SIFT"])

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    i = 0
    precision = []
    recall_all = []
    # read in test image one at a time
    for test_image in test_image_paths:
        test_predicted_icons = []
        test_image_boxes = []
        test_image_array = cv2.imread(test_image, None)
        # segment test image icons (one by one)
        for test_icon_kp, test_icon_desc, icon_path, box in icon_detect_and_compute(
            test_image, params["SIFT"], params["resize_sift_image"], params["resize_segment_icon"]
        ):
            # get each train icon kp and desc
            for train_image, train_icon_name, train_icon_kp, train_icon_desc in train_kp_desc:
                # find any matches between the two
                matches = bf.knnMatch(test_icon_desc, train_icon_desc, k=2)

                valid_matches = []
                for match in matches:
                    if len(match) == 2:
                        m, n = match
                        if m.distance < params["maxRatio"] * n.distance:
                            valid_matches.append([m])

                # need at least 4 for RANSAC to work
                if len(valid_matches) < 4:
                    continue

                train_points = np.float32(
                    [train_icon_kp[m[0].trainIdx].pt for m in valid_matches]
                ).reshape(-1, 1, 2)
                test_points = np.float32(
                    [test_icon_kp[m[0].queryIdx].pt for m in valid_matches]
                ).reshape(-1, 1, 2)

                # return homography matrix and binary matrix of points used in homography
                homography_matrix, binary_matrix = cv2.findHomography(
                    train_points,
                    test_points,
                    cv2.RANSAC,
                    ransacReprojThreshold=params["RNSCThresh"],
                )
                inliers = binary_matrix.ravel().tolist()

                # check if accurate enough match
                if sum(inliers) > params["inliers"]:
                    # split train icon name into number and name
                    icon_num = train_icon_name.split("-")[0]
                    icon_name = "-".join(train_icon_name.split("-")[1:])

                    draw_bounding_box(test_image_array, box, icon_num, icon_name)
                    test_predicted_icons.append(icon_name)
                    test_image_boxes.append(((box[0], box[1]), (box[0] + box[2], box[1] + box[3])))

        if EVAL:
            name = test_image.split("/")
            predicted_filename = name[1] + "-" + name[3][:-4]
            with open(f"./predicted/{predicted_filename}.txt", "w") as f:
                for n in range(len(test_predicted_icons)):
                    f.write(
                        f"{test_predicted_icons[n]}, {str(test_image_boxes[n][0])}, {str(test_image_boxes[n][1])}\n"
                    )
            f.close()

        # Compare predicted to actual icons
        predicted_icons_set = set(test_predicted_icons)
        test_annotated_features = get_test_annotations(test_image_annotations_paths[i])
        actual_icons_set = set([f[0] for f in test_annotated_features])

        true_positive_icons = predicted_icons_set.intersection(actual_icons_set)
        all_icons_len = len(actual_icons_set)

        false_positive_set = predicted_icons_set.difference(actual_icons_set)
        false_negative_set = actual_icons_set.difference(predicted_icons_set)

        recall = len(true_positive_icons) / all_icons_len

        # Print out results
        test_image_name = test_image.split("/")[-1][:-4]
        x = prettytable.PrettyTable(hrules=1)
        x.field_names = ["File: " + test_image_name, "Recall: " + str(recall)]
        x.add_row(
            [
                "True Positives: " + ", ".join(true_positive_icons),
                "False Positives: "
                + (", ".join(false_positive_set) if len(false_positive_set) != 0 else "N/A"),
            ]
        )
        x.add_row(
            [
                "False Negatives: "
                + (", ".join(false_negative_set) if len(false_negative_set) != 0 else "N/A"),
                "True Negatives: N/A",
            ]
        )
        print(x)

        precision.append(
            len(true_positive_icons) / (len(true_positive_icons) + len(false_positive_set))
        )
        recall_all.append(
            len(true_positive_icons) / (len(true_positive_icons) + len(false_negative_set))
        )

        if SHOW:
            cv2.imshow("image", test_image_array)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        i += 1

    if EVAL:
        print(f"Precision: {np.mean(precision)}")
        print(f"Recall: {np.mean(recall_all)}")


tuned_params = {
    "RNSCThresh": 10.759948009735304,
    "SIFT": {
        "contrastThreshold": 0.004139086378011707,
        "edgeThreshold": 17.310486555802317,
        "nOctaveLayers": 7,
        "nfeatures": 1900,
        "sigma": 2.1741346835913578,
    },
    "maxRatio": 0.6779285821510288,
    "inliers": 4,
    "resize_sift_image": 95,
    "resize_segment_icon": 5,
}

main(tuned_params)

if EVAL:
    predicted_test_image_boxes_path = "./predicted/"
    predicted_test_image_boxes = sorted(get_dir_files(predicted_test_image_boxes_path))

    test_image_annotations_paths = sorted(test_image_annotations_paths)

    all_iou = []
    for i in range(len(test_image_annotations_paths)):
        # get and sort ground truth annotations
        annotations_path = test_image_annotations_paths[i]
        test_annotations_list = get_test_annotations(annotations_path)
        test_annotations = sorted(test_annotations_list, key=lambda x: x[0])

        # get and sort predicted annotations
        predicted_annotations_path = predicted_test_image_boxes[i]
        predicted_annotations_list = get_test_annotations(predicted_annotations_path)
        predicted_annotations = sorted(predicted_annotations_list, key=lambda x: x[0])

        for n in range(len(test_annotations)):
            try:
                test_x1, test_y1 = tuple(
                    int(num)
                    for num in test_annotations[n][1]
                    .replace("(", "")
                    .replace(")", "")
                    .replace("...", "")
                    .split(", ")
                )
                test_x2, test_y2 = tuple(
                    int(num)
                    for num in test_annotations[n][2]
                    .replace("(", "")
                    .replace(")", "")
                    .replace("...", "")
                    .split(", ")
                )

                predicted_x1, predicted_y1 = tuple(
                    int(num)
                    for num in predicted_annotations[n][1]
                    .replace("(", "")
                    .replace(")", "")
                    .replace("...", "")
                    .split(", ")
                )
                predicted_x2, predicted_y2 = tuple(
                    int(num)
                    for num in predicted_annotations[n][2]
                    .replace("(", "")
                    .replace(")", "")
                    .replace("...", "")
                    .split(", ")
                )

                iou = calculate_iou(
                    (predicted_x1, predicted_y1, predicted_x2, predicted_y2),
                    (test_x1, test_y1, test_x2, test_y2),
                )
            except:
                iou = 0
            all_iou.append(iou)

    print(f"IoU: {np.average(all_iou)}")
