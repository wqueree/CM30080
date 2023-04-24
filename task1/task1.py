import cv2
import math
from math import fabs, degrees, acos
import numpy as np
import sys
import time
from sklearn.metrics import mean_squared_error
# from hyperopt import tpe, hp, fmin, STATUS_OK, Trials, STATUS_FAIL


def time_convert(sec):
    mins = sec // 60
    sec = sec % 60
    hours = mins // 60
    mins = mins % 60
    print("Time Lapsed = {0}:{1}:{2}".format(int(hours), int(mins), sec))


def preprocess_image(img_path: str, kernel_size):
    """
    - Read image in\n
    - Convert to Grayscale\n
    - Apply a Gaussian Blur to remove noise

    :param kernel_size:
    :param img_path: path to image
    :returns: blurred and grayscale image
    """

    # Read the original image
    img = cv2.imread(img_path)

    # Convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Use a Gaussian Blur on the image to improve edge detection
    img_blur = cv2.GaussianBlur(img_gray, (kernel_size, kernel_size), 0)

    return img_blur


def detect_edges(img, min_edge, max_edge):
    """
    Use Canny Edge Detection to find edges
    :param img: test
    :return: edges
    """
    # Canny Edge Detection
    edges = cv2.Canny(image=img, threshold1=min_edge, threshold2=max_edge)

    #
    dilated_edges = cv2.dilate(edges, np.ones((2, 2), dtype=np.uint8))

    return edges


# Use houghlines to convert edges to points
def detect_lines(path: str, edges: np.array, rho, theta, threshold, min_length, max_gap):
    lines = cv2.HoughLinesP(edges, rho=rho, theta=theta, threshold=threshold, minLineLength=min_length, maxLineGap=max_gap)
    line_list = []
    img = cv2.imread(path)

    for line in lines:
        x1 = line[0][0]
        y1 = line[0][1]
        x2 = line[0][2]
        y2 = line[0][3]
        line_list.append([[x1, y1], [x2, y2]])

        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 1)  # add line to image for representation

    return line_list


# Function of first approach to calculating angle using gradients
def calculate_gradients(line_coords: list):
    tolerance = 0.5
    m_1 = None
    m_2 = None
    for line in line_coords:
        m = (line[1][1] - line[0][1]) / (line[1][0] - line[0][0])
        if not m_1:
            m_1 = m
        elif fabs(m - m_1) < tolerance:  # check if difference is greater than tolerance
            m_1 = (m_1 + m) / 2
        elif not m_2:
            m_2 = m
        elif fabs(m - m_2) < tolerance:
            m_2 = (m_2 + m) / 2

    return m_1, m_2


def calculate_line_segments(line_coords: list):
    """
    Taking three points (line_coords) find the average and return three points.\n
    - The points consist of where the lines intersect and the two ends that do not meet
    """
    # check which x and y coords are similar
    tolerance = 10
    point_1 = line_coords[0][0]  # guaranteed to be two different points
    point_2 = line_coords[0][1]
    point_3 = None
    counters = [0, 0, 0]

    for i in range(1, len(line_coords)):
        x_start = line_coords[i][0][0]
        y_start = line_coords[i][0][1]
        x_end = line_coords[i][1][0]
        y_end = line_coords[i][1][1]

        start = end = False

        #######################################################################
        ## Condition statements to check which points refer with which lines ##
        #######################################################################
        if fabs(x_start - point_1[0]) < tolerance and fabs(
                y_start - point_1[1]) < tolerance:  # case - first set of coords with POINT 1
            point_1[0] = (point_1[0] + x_start) / 2
            point_1[1] = (point_1[1] + y_start) / 2
            start = True
            counters[0] += 1

        if fabs(x_end - point_1[0]) < tolerance and fabs(
                y_end - point_1[1]) < tolerance:  # case - second set of coords with POINT 1
            point_1[0] = (point_1[0] + x_end) / 2
            point_1[1] = (point_1[1] + y_end) / 2
            end = True
            counters[0] += 1

        if fabs(x_start - point_2[0]) < tolerance and fabs(
                y_start - point_2[1]) < tolerance:  # case - first set of coords with POINT 2
            point_2[0] = (point_2[0] + x_start) / 2
            point_2[1] = (point_2[1] + y_start) / 2
            start = True
            counters[1] += 1

        if fabs(x_end - point_2[0]) < tolerance and fabs(
                y_end - point_2[1]) < tolerance:  # case - second set of coords with POINT 2
            point_2[0] = (point_2[0] + x_end) / 2
            point_2[1] = (point_2[1] + y_end) / 2
            end = True
            counters[1] += 1

        if not point_3:  # check if point_3 has been allocated
            if start and end:
                pass
            elif start:
                point_3 = [x_end, y_end]
            elif end:
                point_3 = [x_start, y_start]
        else:
            if fabs(x_start - point_3[0]) < tolerance:  # case - first set of coords with POINT 2
                point_3[0] = (point_3[0] + x_start) / 2
                point_3[1] = (point_3[1] + y_start) / 2
                counters[2] += 1
            elif fabs(x_end - point_3[0]) < tolerance:  # case - second set of coords with POINT 2
                point_3[0] = (point_3[0] + x_end) / 2
                point_3[1] = (point_3[1] + y_end) / 2
                counters[2] += 1

    max_count = max(counters)
    if counters[0] == max_count:
        temp = point_3
        point_3 = point_1
        point_1 = temp
    elif counters[1] == max_count:
        temp = point_3
        point_3 = point_2
        point_2 = temp

    return point_1, point_2, point_3


# Pythagoras Calculation
def calculate_distance(point_1, point_2):
    x_difference = point_2[0] - point_1[0]
    y_difference = point_2[1] - point_1[1]
    # pythagoras to find difference
    return math.sqrt((x_difference ** 2) + (y_difference ** 2))


# Apply Cosine Rule to the 3 points
def cosine_rule(point_1, point_2, point_3, path):
    if not point_3 or not point_1 or not point_2:
        print(f"Not enough lines to calculate angle, please ensure {path} and parameters are correct")
        return -100
    a = calculate_distance(point_1, point_3)
    b = calculate_distance(point_2, point_3)
    c = calculate_distance(point_1, point_2)

    return degrees(acos(((a ** 2) + (b ** 2) - (c ** 2)) / (2 * a * b)))


# Calculate the average accuracy from each image
def calculate_accuracy(true_vals, pred_vals):
    accuracy = []
    for i in range(len(true_vals)):
        accuracy.append(abs((true_vals[i] - pred_vals[i]) / true_vals[i]))

    return np.mean(accuracy)


# Function to apply necessary checks on image list file
def file_name_checks(args):
    if len(args) == 1:
        print("The text file containing image names and angles has not been passed, searching for list.txt by default.")
        return "list.txt"
    file = sys.argv[1]
    if not file.endswith(".txt"):
        print("Please provide file as a .txt type")
        exit()

    return file


def start_method(params):
    min_edge, max_edge, kernel_size, rho, theta, threshold, min_length, max_gap = int(params["min_edge"]), int(params["max_edge"]), int(
        params["kernel_size"]), float(params["rho"]), float(params["theta"]), int(
        params["threshold"]), float(params["min_length"]), int(params["max_gap"])
    start_time = time.time()

    predicted_angle = []
    real_angle = []

    file_name = file_name_checks(sys.argv)
    list_file = open(file_name, "r")

    for image_text in list_file:
        line = image_text.split(',')
        path = line[0]
        correct_angle = line[1].replace('\n', '')

        image = preprocess_image(path, kernel_size)
        edges = detect_edges(image, min_edge, max_edge)
        line_coords = detect_lines(path, edges, rho, theta, threshold, min_length, max_gap)

        p_1, p_2, p_3 = calculate_line_segments(line_coords)

        angle = cosine_rule(p_1, p_2, p_3, path)

        predicted_angle.append(angle)
        real_angle.append(int(float(correct_angle)))

        print(f"{path}, Calculated Angle: {angle}, Actual Angle: {correct_angle}")

    end_time = time.time()

    # MSE Calculation
    print("MSE: ", mean_squared_error(real_angle, predicted_angle))

    # Accuracy Calculation
    accuracy = 1 - calculate_accuracy(real_angle, predicted_angle)
    print("Accuracy: ", accuracy)

    # Time Lapsed Calculation
    time_lapsed = end_time - start_time
    print(f"Time lapsed: {time_lapsed}")


# Our parameters found during tuning
params = {
    "min_edge": 50,
    "max_edge": 150,
    "kernel_size": 5,
    "rho": 0.991,
    "theta": np.pi/180,
    "threshold": 60,
    "min_length": 80,
    "max_gap": 50
}

start_method(params)
