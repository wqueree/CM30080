from image_compare import compare_rgb, compare_rgb_file, compare_grayscale, compare_grayscale_file
import cv2
import numpy as np

def main() -> None:
    image_1 = "C:\\Users\\wquer\\GitHub\\Personal\\CM30080\\task2\\train\\png\\001-lighthouse.png"
    image_2 = "C:\\Users\\wquer\\GitHub\\Personal\\CM30080\\task2\\train\\png\\002-bike.png"
    print(compare_grayscale_file(image_1, image_2))
    print(compare_rgb_file(image_1, image_2))

if __name__ == "__main__":
    main()
    # Get bounding boxes, sort by size, remove all lower than some size threshold