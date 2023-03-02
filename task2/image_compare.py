import cv2
from pathlib import Path

def compare_rgb_path(image_path_1: Path, image_path_2) -> float:
    image_1 = cv2.imread(image_path_1)
    print(type(image_1))
    image_2 = cv2.imread(image_path_2)
    return compare_rgb(image_1, image_2)

def compare_rgb(image_1, image_2) -> float:
    pass

def mse(image_1, image_2) -> float:
    pass


