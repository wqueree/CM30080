import math
from typing import Dict, Tuple

import cv2
import numpy as np
from color import BLACK, WHITE


def generate_templates(icon: np.ndarray, theta_interval: int, sampling_levels: int, sampling_interval: int, kernel_size: int) -> Dict[int, Dict[int, np.ndarray]]:
    if icon.shape[2] != 4:
        raise ValueError("Image must be in BGRA format. Use cv2.IMREAD_UNCHANGED for PNG images.")
    icon_padded = pad_icon(icon, WHITE)
    rotations_masked: Dict[int, np.ndarray] = generate_masked_rotations(icon_padded, theta_interval)
    templates: Dict[int, Dict[int, np.ndarray]] = dict()
    for angle, rotation in rotations_masked.items():
        templates[angle] = generate_gaussian_pyramid(rotation, 5, 5, 5)
    return templates


def generate_gaussian_pyramid(icon: np.ndarray, sampling_levels: int, sampling_interval: int, kernel_size: int) -> Dict[int, np.ndarray]:
    gaussian_pyramid: Dict[int, np.ndarray] = dict()
    gaussian_pyramid[0] = icon
    for i in range(1, sampling_levels):
        icon = cv2.GaussianBlur(icon, (kernel_size, kernel_size), 0)
        icon = np.delete(icon, list(range(0, icon.shape[0], sampling_interval)), axis=0)
        icon = np.delete(icon, list(range(0, icon.shape[1], sampling_interval)), axis=1)
        gaussian_pyramid[i] = icon
    return gaussian_pyramid


def pad_icon(icon: np.ndarray, color: Tuple[int] = WHITE) -> np.ndarray:
    icon_height, icon_width = icon.shape[:2]
    diagonal = math.ceil(math.sqrt((icon_height ** 2) + icon_width ** 2))
    vertical_border_size = math.ceil((diagonal - icon_height) / 2)
    horizontal_border_size = math.ceil((diagonal - icon_width) / 2)
    icon_padded = cv2.copyMakeBorder(icon, vertical_border_size, vertical_border_size, horizontal_border_size, horizontal_border_size, cv2.BORDER_CONSTANT, None, WHITE)
    return icon_padded


def generate_masked_rotations(icon_padded: np.ndarray, theta_interval: int) -> Dict[int, np.ndarray]:
    if 360 % theta_interval != 0:
        raise ValueError(f"Angle {theta_interval} degrees is not divisible by 360.")
    icon_height, icon_width = icon_padded.shape[:2]
    icon_center: Tuple[int] = (icon_width // 2, icon_height // 2)
    masked_rotations: Dict[int, np.ndarray] = dict()
    masked_rotations[0] = mask_icon(icon_padded)
    for theta in range(theta_interval, 360, theta_interval):
        rotation_matrix: np.ndarray = cv2.getRotationMatrix2D(icon_center, theta, 1.0)
        rotated: np.ndarray = cv2.warpAffine(icon_padded, rotation_matrix, (icon_width, icon_height), borderMode=cv2.BORDER_CONSTANT, borderValue=WHITE)
        masked_rotations[theta] = mask_icon(rotated)
    return masked_rotations


def mask_icon(icon_bgra):
    alpha = icon_bgra[:, :, 3]
    alpha_mask = cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR)
    icon_bgr = cv2.cvtColor(icon_bgra, cv2.COLOR_BGRA2BGR)
    icon_masked: np.ndarray = cv2.bitwise_and(icon_bgr, alpha_mask)
    icon_masked = cv2.copyMakeBorder(icon_masked, 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, BLACK)
    return icon_masked
