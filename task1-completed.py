import cv2
import math
from math import fabs, degrees, acos
import numpy as np


def preprocess_image(path):
    # Read the original image
    img = cv2.imread(path)

    # Convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Use a Gaussian Blur on the image to improve edge detection
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

    return img_blur


def detect_edges(img):
    # Canny Edge Detection
    edges = cv2.Canny(image=img, threshold1=50, threshold2=150)

    # dilated_edges = cv2.dilate(edges, np.ones((2, 2), dtype=np.uint8))

    return edges


def detect_lines(path, edges, i):
    lines = cv2.HoughLinesP(edges, rho=0.991, theta=1 * np.pi / 180, threshold=55, minLineLength=80, maxLineGap=50)
    line_list = []
    img = cv2.imread(path)
    for line in lines:
        x1 = line[0][0]
        y1 = line[0][1]
        x2 = line[0][2]
        y2 = line[0][3]
        line_list.append([[x1, y1], [x2, y2]])

        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 1)  # add line to image for representation

    cv2.imwrite(f'houghlines{i}.jpg', img)
    return line_list


def calculate_gradients(line_coords):
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


def calculate_line_segments(line_coords):
    # check which x and y coords are similar
    tolerance = 10
    point_1 = line_coords[0][0]  # guaranteed to be two different points
    point_2 = line_coords[0][1]
    point_3 = None
    start = False
    end = False
    counter_1 = 0
    counter_2 = 0
    counter_3 = 0
    for i in range(1, len(line_coords)):
        x_start = line_coords[i][0][0]
        y_start = line_coords[i][0][1]
        x_end = line_coords[i][1][0]
        y_end = line_coords[i][1][1]
        # print(x_start, y_start, x_end, y_end)
        if fabs(x_start - point_1[0]) < tolerance and fabs(y_start - point_1[1]) < tolerance:  # case - first set of coords with POINT 1
            point_1[0] = (point_1[0] + x_start) / 2
            point_1[1] = (point_1[1] + y_start) / 2
            start = True
            counter_1 += 1
        if fabs(x_end - point_1[0]) < tolerance and fabs(y_end - point_1[1]) < tolerance:  # case - second set of coords with POINT 1
            point_1[0] = (point_1[0] + x_end) / 2
            point_1[1] = (point_1[1] + y_end) / 2
            end = True
            counter_1 += 1
        if fabs(x_start - point_2[0]) < tolerance and fabs(y_start - point_2[1]) < tolerance:  # case - first set of coords with POINT 2
            point_2[0] = (point_2[0] + x_start) / 2
            point_2[1] = (point_2[1] + y_start) / 2
            start = True
            counter_2 += 1
        if fabs(x_end - point_2[0]) < tolerance and fabs(y_end - point_2[1]) < tolerance:  # case - second set of coords with POINT 2
            point_2[0] = (point_2[0] + x_end) / 2
            point_2[1] = (point_2[1] + y_end) / 2
            end = True
            counter_2 += 1

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
                counter_3 += 1
            elif fabs(x_end - point_3[0]) < tolerance:  # case - second set of coords with POINT 2
                point_3[0] = (point_3[0] + x_end) / 2
                point_3[1] = (point_3[1] + y_end) / 2
                counter_3 += 1
        start = False
        end = False
    max_count = max(counter_1, counter_2, counter_3)
    if counter_1 == max_count:
        temp = point_3
        point_3 = point_1
        point_1 = temp
    elif counter_2 == max_count:
        temp = point_3
        point_3 = point_2
        point_2 = temp
    return point_1, point_2, point_3


def calculate_distance(point_1, point_2):
    x_difference = point_2[0] - point_1[0]
    y_difference = point_2[1] - point_1[1]
    # pythagoras to find difference
    return math.sqrt((x_difference ** 2) + (y_difference ** 2))


def cosine_rule(point_1, point_2, point_3):
    a = calculate_distance(point_1, point_3)
    b = calculate_distance(point_2, point_3)
    c = calculate_distance(point_1, point_2)

    return acos(((a ** 2) + (b ** 2) - (c ** 2))/(2 * a * b))


correct_angles = []
list_file = open("list.txt", "r")
for text in list_file:
    angle_text = text.split(',')[1]
    correct_angles.append(angle_text.replace('\n', ''))

for i in range(1, 11):
    path = f"image{i}.png"
    image = preprocess_image(path)
    edges = detect_edges(image)
    line_coords = detect_lines(path, edges, i)

    point_1, point_2, point_3 = calculate_line_segments(line_coords)

    angle = degrees(cosine_rule(point_1, point_2, point_3))

    print(f"Image{i}, Calculated: {int(angle)}, Actual: {correct_angles[i - 1]}")
