import math
from typing import Dict, Tuple

import cv2 as cv
import numpy as np
from color import BLACK, WHITE


def generate_templates(icon: np.ndarray, theta_interval: int, sampling_levels: int, sampling_interval: int, kernel_size: int) -> Dict[int, Dict[int, np.ndarray]]:
    if icon.shape[2] != 4:
        raise ValueError("Image must be in BGRA format. Use cv.IMREAD_UNCHANGED for PNG images.")
    icon_padded = pad_icon(icon, WHITE)
    rotations_masked: Dict[int, np.ndarray] = generate_masked_rotations(icon_padded, theta_interval)
    templates: Dict[int, Dict[int, np.ndarray]] = dict()
    for angle, rotation in rotations_masked.items():
        templates[angle] = generate_gaussian_pyramid(rotation, 5, 5, 5)
    return templates


def generate_gaussian_pyramid(icon: np.ndarray, sampling_levels: int, sampling_interval: int = 2, kernel_size: int = 3) -> Dict[int, np.ndarray]:
    gaussian_pyramid: Dict[int, np.ndarray] = dict()
    gaussian_pyramid[0] = icon
    for i in range(1, sampling_levels):
        icon = cv.pyrDown(icon)
        # icon = cv.GaussianBlur(icon, (kernel_size, kernel_size), 0)
        # icon = np.delete(icon, list(range(0, icon.shape[0], sampling_interval)), axis=0)
        # icon = np.delete(icon, list(range(0, icon.shape[1], sampling_interval)), axis=1)
        gaussian_pyramid[i] = icon
    return gaussian_pyramid


def pad_icon(icon: np.ndarray, color: Tuple[int] = WHITE) -> np.ndarray:
    icon_height, icon_width = icon.shape[:2]
    diagonal = math.ceil(math.sqrt((icon_height ** 2) + icon_width ** 2))
    vertical_border_size = math.ceil((diagonal - icon_height) / 2)
    horizontal_border_size = math.ceil((diagonal - icon_width) / 2)
    icon_padded = cv.copyMakeBorder(icon, vertical_border_size, vertical_border_size, horizontal_border_size, horizontal_border_size, cv.BORDER_CONSTANT, None, WHITE)
    return icon_padded


def generate_masked_rotations(icon_padded: np.ndarray, theta_interval: int) -> Dict[int, np.ndarray]:
    if 360 % theta_interval != 0:
        raise ValueError(f"Angle {theta_interval} degrees is not divisible by 360.")
    icon_height, icon_width = icon_padded.shape[:2]
    icon_center: Tuple[int] = (icon_width // 2, icon_height // 2)
    masked_rotations: Dict[int, np.ndarray] = dict()
    masked_rotations[0] = mask_icon(icon_padded)
    for theta in range(theta_interval, 360, theta_interval):
        rotation_matrix: np.ndarray = cv.getRotationMatrix2D(icon_center, theta, 1.0)
        rotated: np.ndarray = cv.warpAffine(icon_padded, rotation_matrix, (icon_width, icon_height), borderMode=cv.BORDER_CONSTANT, borderValue=WHITE)
        masked_rotations[theta] = mask_icon(rotated)
    return masked_rotations


def mask_icon(icon_bgra):
    alpha = icon_bgra[:, :, 3]
    alpha_mask = cv.cvtColor(alpha, cv.COLOR_GRAY2BGR)
    icon_bgr = cv.cvtColor(icon_bgra, cv.COLOR_BGRA2BGR)
    icon_masked: np.ndarray = cv.bitwise_and(icon_bgr, alpha_mask)
    icon_masked = cv.copyMakeBorder(icon_masked, 1, 1, 1, 1, cv.BORDER_CONSTANT, None, BLACK)
    return icon_masked


def generate_normalized_template(template_gray: np.ndarray) -> np.ndarray:
    template_normalized = cv.normalize(template_gray, dst=None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
    template_normalized = cv.equalizeHist(template_gray)
    cv.imshow("template_gray", template_gray)
    cv.imshow("template_normalized", template_normalized)
    cv.waitKey(0)
    cv.destroyAllWindows()

def naive_match_template(image: np.ndarray, template: np.ndarray) -> Tuple[int]:
    template_mean: int = round(template.mean())
    match_map: np.ndarray = cv.normalize(cv.filter2D(image, ddepth=cv.CV_32F, kernel=template), dst=None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    # cv.imshow("image", image)
    # cv.imshow("naive_template", template)
    # cv.imshow("naive_match_map", match_map)
    _, max_val, _, max_loc = cv.minMaxLoc(match_map)
    return match_map, max_val, max_loc


def zero_mean_match_template(image: np.ndarray, naive_template: np.ndarray) -> Tuple[np.ndarray, int, Tuple[int]]:
    naive_template_mean: int = round(naive_template.mean())
    zero_mean_template: np.ndarray = np.copy(naive_template).astype(np.int8)
    zero_mean_template -= naive_template_mean
    # match_map: np.ndarray = cv.normalize(cv.filter2D(image, ddepth=cv.CV_32F, kernel=zero_mean_template), dst=None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    match_map: np.ndarray = cv.filter2D(image, ddepth=cv.CV_32F, kernel=zero_mean_template)
    # cv.imshow("image", image)
    # cv.imshow("naive_template", naive_template)
    # cv.imshow("zero_mean_template", zero_mean_template)
    # cv.imshow("match_map", match_map)
    _, max_val, _, max_loc = cv.minMaxLoc(match_map)
    return match_map, max_val, max_loc

def ssd_match_template(image: np.ndarray, template: np.ndarray) -> Tuple[np.ndarray, int, Tuple[int]]:
    image_bordered: np.ndarray = cv.copyMakeBorder(image, template.shape[0] // 2, template.shape[0] // 2, template.shape[1] // 2, template.shape[1] // 2, cv.BORDER_DEFAULT, None, 0)
    match_map: np.ndarray = np.zeros((image_bordered.shape[0] - template.shape[0] + 1, image_bordered.shape[1] - template.shape[1] + 1))
    for y in range(match_map.shape[0]):
        for x in range(match_map.shape[1]):
            match_map[y, x] = np.sum((image_bordered[y : y + template.shape[0], x : x + template.shape[1]] - template) ** 2)
    min_val, _, min_loc, _ = cv.minMaxLoc(match_map)
    return match_map, min_val, min_loc


# image = cv.imread("/home/wqueree/Downloads/200.jpg", cv.IMREAD_GRAYSCALE)
# naive_template = image[400:420, 400:420]
# naive_template_mean = round(naive_template.mean())
# zero_mean_template = np.copy(naive_template).astype(np.int8)
# zero_mean_template -= naive_template_mean
# naive_match = cv.normalize(cv.filter2D(image, ddepth=cv.CV_32F, kernel=naive_template), dst=None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
# zero_mean_match = cv.normalize(cv.filter2D(image, ddepth=cv.CV_32F, kernel=zero_mean_template), dst=None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
# cv.imshow("image", image)
# cv.imshow("naive_template", naive_template)
# cv.imshow("zero_mean_template", zero_mean_template)
# cv.imshow("naive_match", naive_match)
# cv.imshow("zero_mean_match", zero_mean_match)


# print(np.unravel_index(np.argmax(naive_match), naive_match.shape)[::-1])
# print(np.unravel_index(np.argmax(zero_mean_match), zero_mean_match.shape)[::-1])

# cv.waitKey(0)
# cv.destroyAllWindows()