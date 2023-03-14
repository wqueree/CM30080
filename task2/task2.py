import cv2
import numpy as np

from bounding_box import get_bounding_boxes, render_bounding_box
from colors import WHITE
from image_compare import compare_rgb
from pathlib import Path
from typing import Dict, List, Tuple


def main() -> None:
    predict_image_directory_path: Path = Path("./predict/images")
    predict_image_directory_path.mkdir(parents=True, exist_ok=True)
    predict_label_directory_path: Path = Path("./predict/labels")
    predict_label_directory_path.mkdir(parents=True, exist_ok=True)
    test_directory_path: Path = Path("./test/images").resolve(strict=True)
    train_directory_path: Path = Path("./train/png").resolve(strict=True)
    icons: Dict[str, np.ndarray] = get_train_icons(train_directory_path)
    for image_path in test_directory_path.glob("*.png"):
        image: np.ndarray = cv2.imread(str(image_path))
        predict_icon_classes(image, image_path.name, icons, predict_image_directory_path, predict_label_directory_path)


def predict_icon_classes(image: np.ndarray, image_name: str, train_icons: Dict[str, np.ndarray], predict_image_directory_path: Path, predict_label_directory_path: Path) -> str:
    """Predicts icon classes for all icons in the given image."""
    bounding_boxes: List[Tuple[int, float]] = get_bounding_boxes(image, min_contour_area=1500, max_contour_area=100000)
    test_icons: List[np.ndarray] = get_image_icons(image, bounding_boxes)
    image_predict: np.ndarray = np.copy(image)
    for bounding_box, test_icon in zip(bounding_boxes, test_icons):
        icon_errors: Dict[str, float] = dict()
        for class_name, train_icon in train_icons.items():
            scaled_train_icon: np.ndarray = cv2.resize(train_icon, test_icon.shape[1::-1], interpolation=cv2.INTER_AREA)
            icon_errors[class_name] = compare_rgb(scaled_train_icon, test_icon)
        prediction: str = min(icon_errors, key=icon_errors.get)
        label_top, label_bottom = prediction.split("-", maxsplit=1)
        render_bounding_box(image_predict, bounding_box, label_top=label_top, label_bottom=label_bottom)  
    cv2.imwrite(f"{predict_image_directory_path}/{image_name}", image_predict)


def get_train_icons(train_directory_path: Path) -> Dict[str, np.ndarray]:
    """Gets and crops training icons from the given directory."""
    cropped_icons: Dict[str, np.ndarray] = dict()
    for icon_path in train_directory_path.glob("*.png"):
        # Read icon and pad with 1px white border
        icon: np.ndarray = cv2.imread(str(icon_path))
        icon = cv2.copyMakeBorder(icon, 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, WHITE)
        # Crop icons
        x, y, w, h = get_bounding_boxes(icon, min_contour_area=100000, max_contour_area=275000)[0]
        icon_crop: np.ndarray = icon[y:y + h, x:x + w]
        cropped_icons[icon_path.stem] = icon_crop
    return cropped_icons


def get_image_icons(image: np.ndarray, bounding_boxes: List[Tuple[int, float]]):
    """Gets and crops icons from the given image by the provided bounding boxes."""
    cropped_icons: List[np.ndarray] = list()
    for x, y, w, h in bounding_boxes:
        icon_crop: np.ndarray = image[y:y + h, x:x + w]
        cropped_icons.append(icon_crop)
    return cropped_icons


if __name__ == "__main__":
    main()
