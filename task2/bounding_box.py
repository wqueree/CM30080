import cv2
import numpy as np

from colors import RED
from pathlib import Path
from typing import List, Union, Tuple


def get_bounding_boxes_file(image_path: Union[Path, str], min_contour_area: int, max_contour_area: int) -> List[Tuple[int, float]]:
    image: np.ndarray = cv2.imread(str(image_path))
    return get_bounding_boxes(image, min_contour_area, max_contour_area)


def get_bounding_boxes(image: np.ndarray, min_contour_area: int, max_contour_area: int) -> List[Tuple[int, float]]:
    bounding_boxes: List[Tuple[int, float]] = list()
    image_greyscale: np.ndarray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image_binary = cv2.threshold(image_greyscale, 240, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(image_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_areas: List[float] = list()
    for i, contour in enumerate(contours):
        contour_area: float = cv2.contourArea(contour)
        if min_contour_area < contour_area < max_contour_area:
            bounding_boxes.append(cv2.boundingRect(contour))
            contour_areas.append(contour_area)
    sorted_bounding_boxes: List[Tuple[int, float]] = [bounding_box for _, bounding_box in sorted(zip(contour_areas, bounding_boxes))]
    return sorted_bounding_boxes


def render_bounding_boxes(image: np.ndarray, bounding_boxes: List[Tuple[int, float]], color: Tuple[int] = RED) -> None:
    for bounding_box in bounding_boxes:
        render_bounding_box(image, bounding_box)


def render_bounding_box(image: np.ndarray, bounding_box: Tuple[int, float], color: Tuple[int] = RED, label_top: str = "", label_bottom: str = "") -> None:
    x, y, w, h = bounding_box
    cv2.rectangle(image, (x, y), (x + w, y + h), RED, 1)
    cv2.putText(image, label_top, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 1)
    cv2.putText(image, label_bottom, (x, y + h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 1)
