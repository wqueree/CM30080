import cv2
import numpy as np


def pad_image(x_1, y_1, x_2, y_2, width, height):
    x = max(x_1 - 1, 0)
    y = max(y_1 - 1, 0)
    w = min(x_2 + 1, height) - x
    h = min(y_2 + 1, width) - y
    return x, y, w, h


def get_rotated_box(contour, scale):
    (center, size, angle) = cv2.minAreaRect(contour)

    # Change the variables to original size
    center = tuple(np.array(center) / scale)
    size = tuple(np.array(size) / scale)

    # Convert the box to a 4-point bounding box
    box = cv2.boxPoints((center, size, angle))
    box = np.int0(box)
    return box


def get_icon_boxes(image, resize_segment_icon, show=False):
    width, height = image.shape[:2]
    resized_image = cv2.resize(image, None, fx=resize_segment_icon, fy=resize_segment_icon, interpolation=cv2.INTER_LINEAR)
    blurred_image = cv2.GaussianBlur(resized_image, (5, 5), 0)

    # Apply binary thresholding to create a binary image
    ret, thresh = cv2.threshold(blurred_image, 240, 255, cv2.THRESH_BINARY_INV)

    # Apply morphological opening to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morph_open = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Perform morphological closing to remove small holes in the objects
    morph_closed = cv2.morphologyEx(morph_open, cv2.MORPH_CLOSE, kernel)

    # Find contours in the image
    contours, hierarchy = cv2.findContours(morph_closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    bounding_coords = []
    for cnt in contours:
        x, y, w, h = [round(i / resize_segment_icon) for i in cv2.boundingRect(cnt)]
        # some contours were too small so best to remove
        if w * h < 500:
            continue
        x, y, w, h = pad_image(x, y, x + w, y + h, width, height)
        bounding_coords.append(((x, y, w, h), get_rotated_box(cnt, resize_segment_icon)))

    # check if a bounding box is contained completely within another
    i = 0
    while i < len(bounding_coords):
        box_1 = bounding_coords[i][0]
        # check all coords are smaller than another
        n = 0
        while n < len(bounding_coords):
            box_2 = bounding_coords[n][0]
            if box_1[0] < box_2[0] < box_1[0] + box_1[2] and box_1[1] < box_2[1] < box_1[1] + box_1[3]:
                bounding_coords.pop(n)
                continue
            n += 1
        i += 1

    if show:
        # Draw bounding boxes around the contours
        for i in range(0, len(bounding_coords)):
            (x, y, w, h), rotated = bounding_coords[i]
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the result
        cv2.imshow('Result', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return bounding_coords
