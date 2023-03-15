import cv2
import math
import numpy as np

from colors import WHITE
from typing import Dict, Tuple


def generate_templates(icon: np.ndarray, theta_interval: int) -> Dict[int, Dict[int, np.ndarray]]:
    if 360 % theta_interval != 0:
        raise ValueError(f"Angle {theta_interval} degrees is not divisible by 360.")
    
    # Pad for rotation
    icon_height, icon_width = icon.shape[:2]
    diagonal = math.ceil(math.sqrt((icon_height ** 2) + icon_width ** 2))
    vertical_border_size = math.ceil((diagonal - icon_height) / 2)
    horizontal_border_size = math.ceil((diagonal - icon_width) / 2)
    cv2.imshow("icon", icon)
    icon = cv2.copyMakeBorder(icon, vertical_border_size, vertical_border_size, horizontal_border_size, horizontal_border_size, cv2.BORDER_CONSTANT, None, WHITE)
    icon_height, icon_width = icon.shape[:2]

    # Compute Rotations
    icon_center: Tuple[int] = (icon_width // 2, icon_height // 2)
    rotations: Dict[int, Dict[int, np.ndarray]] = dict()
    rotations[0] = icon
    previous_rotation: np.ndarray = icon
    rotation_matrix: np.ndarray = cv2.getRotationMatrix2D(icon_center, theta_interval, 1.0)
    for theta in range(theta_interval, 360, theta_interval):
        previous_rotation = cv2.warpAffine(previous_rotation, rotation_matrix, (icon_width, icon_height), borderMode=cv2.BORDER_CONSTANT, borderValue=WHITE)
        rotations[theta] = previous_rotation
    return rotations
