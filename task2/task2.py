import cv2
import numpy as np

from bounding_box import get_bounding_boxes, get_bounding_boxes_file, render_bounding_boxes
from image_compare import compare_rgb, compare_rgb_file, compare_grayscale, compare_grayscale_file
from pathlib import Path
from typing import Dict, List, Tuple


def main() -> None:
    test_directory_path: Path = Path("./test/images").resolve(strict=True)
    train_directory_path: Path = Path("./train/png").resolve(strict=True)
    
    for image_path in test_directory_path.glob("*.png"):
        image: np.ndarray = cv2.imread(str(image_path))
        predict_icon_classes(image, train_directory_path)


def predict_icon_classes(image: np.ndarray, train_directory_path: Path) -> str:
    bounding_boxes: List[Tuple[int, float]] = get_bounding_boxes(image)
    for bounding_box in bounding_boxes:
        # TODO Crop images using bboxes
        icon_errors: Dict[str, float] = dict()
        for icon_path in train_directory_path.glob("*.png"):
            icon = cv2.imread(str(icon_path))
            # TODO Mask away background
            # TODO Do comparison

    # cv2.imshow("image", image)
    # render_bounding_boxes(image, bounding_boxes)
    # cv2.imshow("bounding_boxes", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
