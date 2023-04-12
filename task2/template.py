from pathlib import Path
from typing import Dict, List, Tuple

import cv2 as cv
import numpy as np
from color import BLACK

def generate_templates(train_directory_path: Path) -> Dict[str, Dict[int, np.ndarray]]: # Filename -> Sampling Level -> Template
    """Gets and crops training icons from the given directory."""
    icon_templates: Dict[str, Dict[int, np.ndarray]] = dict()
    for icon_path in train_directory_path.glob("*.png"):
        icon_bgra: np.ndarray = cv.imread(str(icon_path), cv.IMREAD_UNCHANGED)
        masked_icon: np.ndarray = mask_icon(icon_bgra)
        masked_icon_grayscale = cv.cvtColor(masked_icon, cv.COLOR_BGR2GRAY)
        icon_templates[icon_path.stem] = generate_gaussian_pyramid(masked_icon_grayscale, 6)
    return icon_templates


def generate_gaussian_pyramid(icon: np.ndarray, sampling_levels: int) -> List[np.ndarray]:
    """Generates a Gaussian pyramid of the given icon with the given number of sampling levels."""
    gaussian_pyramid: List[np.ndarray] = [icon]
    for _ in range(sampling_levels - 1):
        icon = cv.pyrDown(icon)
        gaussian_pyramid.append(icon)
    return gaussian_pyramid


def mask_icon(icon_bgra):
    """Masks the given icon with its alpha channel."""
    alpha = icon_bgra[:, :, 3]
    alpha_mask = cv.cvtColor(alpha, cv.COLOR_GRAY2BGR)
    icon_bgr = cv.cvtColor(icon_bgra, cv.COLOR_BGRA2BGR)
    icon_masked: np.ndarray = cv.bitwise_and(icon_bgr, alpha_mask)
    icon_masked = cv.copyMakeBorder(icon_masked, 1, 1, 1, 1, cv.BORDER_CONSTANT, None, BLACK)
    return icon_masked


def naive_match_template(image: np.ndarray, template: np.ndarray) -> Tuple[int]:
    """Finds the best match of the given template in the given image using the naive method."""
    match_map: np.ndarray = cv.normalize(cv.filter2D(image, ddepth=cv.CV_32F, kernel=template), dst=None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    _, max_val, _, max_loc = cv.minMaxLoc(match_map)
    return max_val, max_loc


def zero_mean_match_template(image: np.ndarray, naive_template: np.ndarray) -> Tuple[np.ndarray, int, Tuple[int]]:
    """Finds the best match of the given template in the given image using the zero-mean method."""
    naive_template_mean: int = round(naive_template.mean())
    zero_mean_template: np.ndarray = np.copy(naive_template).astype(np.int8)
    zero_mean_template -= naive_template_mean
    match_map: np.ndarray = cv.filter2D(image, ddepth=cv.CV_64F, kernel=zero_mean_template)
    _, max_val, _, max_loc = cv.minMaxLoc(match_map)
    return max_val, max_loc


def ssd_match_template(image: np.ndarray, template: np.ndarray) -> Tuple[np.ndarray, int, Tuple[int]]:
    """Finds the best match of the given template in the given image using the sum of squared differences method."""
    image_bordered: np.ndarray = cv.copyMakeBorder(image, template.shape[0] // 2, template.shape[0] // 2, template.shape[1] // 2, template.shape[1] // 2, cv.BORDER_DEFAULT, None, 0)
    match_map: np.ndarray = np.zeros((image_bordered.shape[0] - template.shape[0] + 1, image_bordered.shape[1] - template.shape[1] + 1))
    for y in range(match_map.shape[0]):
        for x in range(match_map.shape[1]):
            match_map[y, x] = np.sum((image_bordered[y : y + template.shape[0], x : x + template.shape[1]] - template) ** 2)
    min_val, _, min_loc, _ = cv.minMaxLoc(match_map)
    return min_val, min_loc
