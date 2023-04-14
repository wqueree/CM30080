from typing import Tuple

import cv2 as cv
import numpy as np


def naive_correlation(image: np.ndarray, template: np.ndarray) -> Tuple[float, Tuple[int, int]]:
    """Finds the best match of the given template in the given image using the naive method."""
    correlation_map: np.ndarray = cv.normalize(
        cv.filter2D(image, ddepth=cv.CV_32F, kernel=template),
        dst=None,
        alpha=0,
        beta=255,
        norm_type=cv.NORM_MINMAX,  # type: ignore
        dtype=cv.CV_8U,
    )
    _, max_val, _, max_loc = cv.minMaxLoc(correlation_map)
    return max_val, max_loc


def zero_mean_correlation(
    image: np.ndarray, naive_template: np.ndarray
) -> Tuple[float, Tuple[int, int]]:
    """Finds the best match of the given template in the given image using the zero-mean method."""
    naive_template_mean: int = round(naive_template.mean())
    zero_mean_template: np.ndarray = np.copy(naive_template).astype(np.int8)
    zero_mean_template -= naive_template_mean
    correlation_map: np.ndarray = cv.filter2D(image, ddepth=cv.CV_64F, kernel=zero_mean_template)
    _, max_val, _, max_loc = cv.minMaxLoc(correlation_map)
    return max_val, max_loc


def ssd_correlation(image: np.ndarray, template: np.ndarray) -> Tuple[float, Tuple[int, int]]:
    """Finds the best match of the given template in the given image using the sum of squared differences method."""
    image_bordered: np.ndarray = cv.copyMakeBorder(
        image,
        template.shape[0] // 2,
        template.shape[0] // 2,
        template.shape[1] // 2,
        template.shape[1] // 2,
        cv.BORDER_DEFAULT,
        None,  # type: ignore
        0,
    )
    correlation_map: np.ndarray = np.zeros(
        (
            image_bordered.shape[0] - template.shape[0] + 1,
            image_bordered.shape[1] - template.shape[1] + 1,
        )
    )
    for y in range(correlation_map.shape[0]):
        for x in range(correlation_map.shape[1]):
            correlation_map[y, x] = np.sum(
                (image_bordered[y : y + template.shape[0], x : x + template.shape[1]] - template)
                ** 2
            )
    min_val, _, min_loc, _ = cv.minMaxLoc(correlation_map)
    return min_val, min_loc
