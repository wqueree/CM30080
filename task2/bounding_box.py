import cv2
import numpy as np

from pathlib import Path
from typing import List, Union, Tuple


def get_bounding_boxes_file(image_path: Union[Path, str], min_contour_area: int = 1500, max_contour_area: int = 100000) -> List[Tuple[int, float]]:
    image: np.ndarray = cv2.imread(str(image_path))
    return get_bounding_boxes(image, min_contour_area, max_contour_area)


def get_bounding_boxes(image: np.ndarray, min_contour_area: int = 1500, max_contour_area: int = 100000) -> List[Tuple[int, float]]:
    bounding_boxes: List[Tuple[int, float]] = list()
    image_greyscale: np.ndarray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image_binary = cv2.threshold(image_greyscale, 240, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(image_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        contour_area: float = cv2.contourArea(contour)
        if min_contour_area < contour_area < max_contour_area:
            bounding_boxes.append(cv2.boundingRect(contour))
    return bounding_boxes


def render_bounding_boxes(image: np.ndarray, bounding_boxes: List[Tuple[int, float]], color: Tuple[int] = (0, 0, 255)) -> None:
    for x, y, w, h in bounding_boxes:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 1)
