from pathlib import Path
from typing import List, Tuple

import cv2 as cv
import numpy as np


def generate_test_images(test_directory_path: Path) -> List[Tuple[str, np.ndarray]]:
    """Generates a list of masked test images from the given test directory path."""
    images: List[Tuple[str, np.ndarray]] = list()
    for image_path in test_directory_path.glob("*.png"):
        image: np.ndarray = cv.imread(str(image_path))
        image_grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        _, image_binary = cv.threshold(image_grayscale, 240, 255, cv.THRESH_BINARY)
        mask = cv.cvtColor(cv.bitwise_not(image_binary), cv.COLOR_GRAY2BGR)
        image_masked = cv.bitwise_and(image, mask)
        image_masked_grayscale = cv.cvtColor(image_masked, cv.COLOR_BGR2GRAY)
        images.append((image, image_masked_grayscale, image_path.name))
    return images
