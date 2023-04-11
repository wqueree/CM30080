from pathlib import Path
from typing import Iterable, List, Tuple, Union

import cv2
import numpy as np
from color import RED


def get_bounding_boxes_file(image_path: Union[Path, str], min_contour_area: int, max_contour_area: int) -> List[Tuple[int, float]]:
    """Gets bounding boxes around icons from an image at the given path."""
    image: np.ndarray = cv2.imread(str(image_path))
    return get_bounding_boxes(image, min_contour_area, max_contour_area)


def get_bounding_boxes(image: np.ndarray, min_contour_area: int, max_contour_area: int, threshold: int) -> List[Tuple[int, float]]:
    """Gets bounding boxes around icons from an image."""
    bounding_boxes: List[Tuple[int, float]] = list()
    image_greyscale: np.ndarray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image_binary = cv2.threshold(image_greyscale, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(image_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_areas: List[float] = list()
    for contour in contours:
        contour_area: float = cv2.contourArea(contour)
        if min_contour_area < contour_area < max_contour_area:
            bounding_boxes.append(cv2.boundingRect(contour))
            contour_areas.append(contour_area)
    sorted_bounding_boxes: List[Tuple[int, float]] = [bounding_box for _, bounding_box in sorted(zip(contour_areas, bounding_boxes))]
    return sorted_bounding_boxes


def render_bounding_boxes(image: np.ndarray, bounding_boxes: Iterable[Tuple[int, str, float]], color: Tuple[int] = RED) -> None:
    """Renders bounding boxes onto the specified image."""
    for bounding_box in bounding_boxes:
        render_bounding_box(image, bounding_box)


def render_bounding_box(image: np.ndarray, bounding_box: Tuple[int, str, float], color: Tuple[int] = RED) -> None:
    """Renders a single labelled bounding box onto the specified image."""
    x, y, w, h, label_top, label_bottom, _ = bounding_box
    cv2.rectangle(image, (x, y), (x + w, y + h), RED, 1)
    cv2.putText(image, label_top, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 1)
    cv2.putText(image, label_bottom, (x, y + h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 1)
