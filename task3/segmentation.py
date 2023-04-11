import cv2


def get_icon_boxes(path):
    # Load the image
    img = cv2.imread(path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply binary thresholding to create a binary image
    ret, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # Apply morphological opening to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Perform morphological closing to remove small holes in the objects
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Find contours in the image
    contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    bounding_coords = []
    for cnt in contours:
        bounding_coords.append(cv2.boundingRect(cnt))

    # check if a bounding box is within another
    i = 0
    while i < len(bounding_coords):
        box_1 = bounding_coords[i]
        # check all coords are smaller than another
        n = 0
        while n < len(bounding_coords):
            box_2 = bounding_coords[n]
            if box_1[0] < box_2[0] < box_1[0] + box_1[2] and box_1[1] < box_2[1] < box_1[1] + box_1[3]:
                bounding_coords.pop(n)
                continue
            n += 1
        i += 1

    # Draw bounding boxes around the contours
    for box in bounding_coords:
        x, y, w, h = box
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the result
    cv2.imshow('Result', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return bounding_coords


# print(segment_icons("./TestWithoutRotations/images/test_image_1.png"))
