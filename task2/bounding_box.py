from pathlib import Path
from typing import Iterable, List, Tuple, Union

import cv2
import numpy as np
from color import RED


def render_bounding_boxes(image: np.ndarray, bounding_boxes: Iterable[Tuple[int, int, int, int, str, str, float]], color: Tuple[int, int, int] = RED) -> None:
    """Renders bounding boxes onto the specified image."""
    for bounding_box in bounding_boxes:
        render_bounding_box(image, bounding_box)


def render_bounding_box(image: np.ndarray, bounding_box: Tuple[int, int, int, int, str, str, float], color: Tuple[int, int, int] = RED) -> None:
    """Renders a single labelled bounding box onto the specified image."""
    x, y, w, h, label_top, label_bottom, _ = bounding_box
    cv2.rectangle(image, (x, y), (x + w, y + h), RED, 1)
    cv2.putText(image, label_top, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 1)
    cv2.putText(image, label_bottom, (x, y + h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 1)
