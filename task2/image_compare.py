import cv2
import numpy as np
from pathlib import Path
from typing import Union


def compare_rgb_file(image_path_1: Union[Path, str], image_path_2: Union[Path, str]) -> float:
    """Returns the mean squared error of two rgb image paths across all channels."""
    image_1 = cv2.imread(str(image_path_1))
    image_2 = cv2.imread(str(image_path_2))
    return compare_rgb(image_1, image_2)


def compare_grayscale_file(image_path_1: Union[Path, str], image_path_2: Union[Path, str]) -> float:
    """Returns the mean squared error of two rgb image paths which are then converted to a single grayscale channel."""
    image_1 = cv2.imread(str(image_path_1))
    image_2 = cv2.imread(str(image_path_2))
    image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
    image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
    return compare_grayscale(image_1, image_2)


def compare_rgb(image_1: np.ndarray, image_2: np.ndarray) -> float:
    """Returns the mean squared error of two rgb images across all channels."""
    error: float = 0.0
    # Calculate Blue Channel MSE
    blue_1: np.ndarray = image_1[:, :, 0]
    blue_2: np.ndarray = image_2[:, :, 0]
    error += mse(blue_1, blue_2)
    # Calculate Green Channel MSE
    green_1: np.ndarray = image_1[:, :, 1]
    green_2: np.ndarray = image_2[:, :, 1]
    error += mse(green_1, green_2)
    # Calculate Red Channel MSE
    red_1: np.ndarray = image_1[:, :, 2]
    red_2: np.ndarray = image_2[:, :, 2]
    error += mse(red_1, red_2)
    return error


def compare_grayscale(image_1: np.ndarray, image_2: np.ndarray) -> float:
    """Returns the mean squared error of two rgb images which are then converted to a single grayscale channel."""
    return mse(image_1, image_2)


def mse(channel_1: np.ndarray, channel_2: np.ndarray) -> float:
    """Returns the mean squared error between two image channels."""
    image_height, image_width = channel_1.shape
    difference = cv2.subtract(channel_1, channel_2)
    error = np.sum(difference ** 2)
    return error
