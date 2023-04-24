from pathlib import Path
from typing import Dict, List

import cv2 as cv
import numpy as np
from color import BLACK


def generate_templates(
    train_directory_path: Path,
    sampling_levels: List[int] = [3],
) -> Dict[str, Dict[int, np.ndarray]]:
    """Gets and crops training icons from the given directory."""
    icon_templates: Dict[str, Dict[int, np.ndarray]] = dict()
    for icon_path in train_directory_path.glob("*.png"):
        icon_bgra: np.ndarray = cv.imread(str(icon_path), cv.IMREAD_UNCHANGED)
        masked_icon: np.ndarray = mask_icon(icon_bgra)
        masked_icon_grayscale = cv.cvtColor(masked_icon, cv.COLOR_BGR2GRAY)
        icon_templates[icon_path.stem] = generate_gaussian_pyramid(
            masked_icon_grayscale, sampling_levels
        )
    return icon_templates


def generate_gaussian_pyramid(
    icon: np.ndarray, sampling_levels: List[int]
) -> Dict[int, np.ndarray]:
    """Generates a Gaussian pyramid of the given icon with the given number of sampling levels."""
    gaussian_pyramid: Dict[int, np.ndarray] = dict()
    for sampling_level in range(1, max(sampling_levels) + 1):
        icon = cv.pyrDown(icon)
        if sampling_level in sampling_levels:
            gaussian_pyramid[sampling_level] = icon
    return gaussian_pyramid


def mask_icon(icon_bgra):
    """Masks the given icon with its alpha channel."""
    alpha = icon_bgra[:, :, 3]
    alpha_mask = cv.cvtColor(alpha, cv.COLOR_GRAY2BGR)
    icon_bgr = cv.cvtColor(icon_bgra, cv.COLOR_BGRA2BGR)
    icon_masked: np.ndarray = cv.bitwise_and(icon_bgr, alpha_mask)
    icon_masked = cv.copyMakeBorder(icon_masked, 1, 1, 1, 1, cv.BORDER_CONSTANT, None, BLACK)  # type: ignore
    return icon_masked
