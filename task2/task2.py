from image_compare import compare_rgb, compare_rgb_file, compare_grayscale, compare_grayscale_file
import cv2
import numpy as np

def main() -> None:
    image_1_path = "C:\\Users\\wquer\\GitHub\\Personal\\CM30080\\task2\\test\\images\\test_image_1.png"
    image_1 = cv2.imread(image_1_path)
    image_1_copy = np.copy(image_1)

    image_1_grey = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
    _, image_1_240 = cv2.threshold(image_1_grey, 240, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(image_1_240, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_areas = [0] * len(contours)

    for i, contour in enumerate(contours):
        contour_area = cv2.contourArea(contour)

        if 20 < contour_area < 100000:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image_1, (x, y), (x + w, y + h), (0, 0, 255), 1)

    cv2.imshow("image_1", image_1)
    cv2.imshow("image_1_grey", image_1_grey)
    cv2.imshow("image_1_240", image_1_240)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # TODO Crop images using bboxes
    # TODO Mask away background
    # TODO Do comparison


if __name__ == "__main__":
    main()
