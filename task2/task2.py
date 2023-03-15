import cv2
import numpy as np

from bounding_box import get_bounding_boxes, render_bounding_box
from colors import BLACK
from image_compare import compare_rgb
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple


# TODO Investigate Nyquist Limit


def main() -> None:
    predict_image_directory_path: Path = Path("./predict/images")
    predict_image_directory_path.mkdir(parents=True, exist_ok=True)
    predict_label_directory_path: Path = Path("./predict/labels")
    predict_label_directory_path.mkdir(parents=True, exist_ok=True)
    test_directory_path: Path = Path("./test/images").resolve(strict=True)
    train_directory_path: Path = Path("./train/png").resolve(strict=True)
    icons: Dict[str, np.ndarray] = get_train_icons(train_directory_path)
    images: List[str, np.ndarray] = get_test_images(test_directory_path)
    for image, image_masked, image_name in tqdm(images):
        predict_icon_classes(image_masked, image, image_name, icons, predict_image_directory_path, predict_label_directory_path)


def predict_icon_classes(image_masked: np.ndarray, image_predict: np.ndarray, image_name: str, train_icons: Dict[str, np.ndarray], predict_image_directory_path: Path, predict_label_directory_path: Path) -> str:
    """Predicts icon classes for all icons in the given image."""
    bounding_boxes: List[Tuple[int, float]] = get_bounding_boxes(image_masked, 1500, 250000, 15)
    test_icons: List[np.ndarray] = get_image_icons(image_masked, bounding_boxes)
    for bounding_box, test_icon in zip(bounding_boxes, test_icons):
        icon_errors: Dict[str, float] = dict()
        cv2.imshow("test_icon", test_icon)
        for class_name, train_icon in train_icons.items():
            scaled_train_icon: np.ndarray = cv2.resize(train_icon, test_icon.shape[1::-1], interpolation=cv2.INTER_AREA)
            mse: float = compare_rgb(scaled_train_icon, test_icon)
            print(f"{class_name} mse: {mse}")
            cv2.imshow("train_icon", scaled_train_icon)
            cv2.waitKey()
            cv2.destroyWindow("train_icon")
            icon_errors[class_name] = mse
        prediction: str = min(icon_errors, key=icon_errors.get)
        label_top, label_bottom = prediction.split("-", maxsplit=1)
        render_bounding_box(image_predict, bounding_box, label_top=label_top, label_bottom=label_bottom)  
    cv2.imwrite(f"{predict_image_directory_path}/{image_name}", image_predict)


def get_train_icons(train_directory_path: Path) -> Dict[str, Dict[int, Dict[int, np.ndarray]]]: # Filename -> Rotation -> Sampling Level
    """Gets and crops training icons from the given directory."""
    cropped_icons: Dict[str, np.ndarray] = dict()
    for icon_path in train_directory_path.glob("*.png"):
        # Read icon and pad with a 1px black border
        icon_bgra: np.ndarray = cv2.imread(str(icon_path), cv2.IMREAD_UNCHANGED)
        alpha = icon_bgra[:, :, 3]
        alpha_mask = cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR)
        icon_bgr = cv2.cvtColor(icon_bgra, cv2.COLOR_BGRA2BGR)
        icon: np.ndarray = cv2.bitwise_and(icon_bgr, alpha_mask)
        icon = cv2.copyMakeBorder(icon, 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, BLACK)
        # Crop icons
        x, y, w, h = get_bounding_boxes(icon, 60000, 275000, 15)[0]
        icon_crop: np.ndarray = icon[y:y + h, x:x + w]
        cropped_icons[icon_path.stem] = icon_crop
    return cropped_icons


def get_test_images(test_directory_path: Path) -> List[Tuple[str, np.ndarray]]:
    images: List[Tuple[str, np.ndarray]] = list()
    for image_path in test_directory_path.glob("*.png"):
        image: np.ndarray = cv2.imread(str(image_path))
        image_greyscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, image_binary = cv2.threshold(image_greyscale, 240, 255, cv2.THRESH_BINARY)
        mask = cv2.cvtColor(cv2.bitwise_not(image_binary), cv2.COLOR_GRAY2BGR)
        image_masked = cv2.bitwise_and(image, mask)
        images.append((image, image_masked, image_path.name))
    return images


def get_image_icons(image: np.ndarray, bounding_boxes: List[Tuple[int, float]]):
    """Gets and crops icons from the given image by the provided bounding boxes."""
    cropped_icons: List[np.ndarray] = list()
    for x, y, w, h in bounding_boxes:
        icon_crop: np.ndarray = image[y:y + h, x:x + w]
        cropped_icons.append(icon_crop)
    return cropped_icons


if __name__ == "__main__":
    main()
