from pathlib import Path
from typing import Dict, List, Set, Tuple

import cv2 as cv
import numpy as np
from bounding_box import get_bounding_boxes, render_bounding_box, render_bounding_boxes
from color import BLACK
from image_compare import compare_rgb
from template import generate_templates, mask_icon, generate_gaussian_pyramid, zero_mean_match_template, ssd_match_template
from tqdm import tqdm

THRESHOLD: int = 165000

# TODO Implement intensity normalization
# TODO Implement normalized cross correlation
# TODO Investigate Nyquist Limit


def main() -> None:
    predict_image_directory_path: Path = Path("./predict/images")
    predict_image_directory_path.mkdir(parents=True, exist_ok=True)
    predict_annotation_directory_path: Path = Path("./predict/annotations")
    predict_annotation_directory_path.mkdir(parents=True, exist_ok=True)
    test_directory_path: Path = Path("./test/images").resolve(strict=True)
    train_directory_path: Path = Path("./train/png").resolve(strict=True)
    templates: Dict[str, Dict[int, np.ndarray]] = get_templates(train_directory_path)
    # for class_name, sampling_levels in templates.items():
    #     for sampling_level, template in sampling_levels.items():
    #         cv.imshow(f"{class_name} {sampling_level}", template)
    #         cv.waitKey()
    #         cv.destroyWindow(f"{class_name} {sampling_level}")
    images: List[str, np.ndarray] = get_test_images(test_directory_path)
    progress: tqdm = tqdm(total=len(images) * len(templates), desc="Predicting Classes")
    for image, image_masked, image_name in images:
    #     cv.imshow("image", image)
    #     cv.imshow("image_masked", image_masked)
    #     cv.waitKey()
    #     cv.destroyAllWindows()
        predict_icon_classes(image_masked, image, image_name, templates, predict_image_directory_path, predict_annotation_directory_path, progress)
    cv.waitKey()
    cv.waitKey()
    cv.destroyAllWindows()


def predict_icon_classes(image_masked: np.ndarray, image_predict: np.ndarray, image_name: str, templates: Dict[str, np.ndarray], predict_image_directory_path: Path, predict_annotation_directory_path: Path, progress: tqdm) -> str:
    """Predicts icon classes for all icons in the given image."""
    # cv.imshow(image_name, image_masked)
    bounding_boxes: Set[Tuple[int]] = set()
    for class_name, sampling_levels in templates.items():
        # for sampling_level, template in sampling_levels.items():
        template = sampling_levels[3]
        if template.shape[0] < image_masked.shape[0] and template.shape[1] < image_masked.shape[1]:
            match_map, pred_value, pred_centre = ssd_match_template(image_masked, template)
            if pred_value < THRESHOLD:
                label_top, label_bottom = class_name.split("-", maxsplit=1)
                bounding_box_origin: Tuple[int, int] = (pred_centre[0] - 31, pred_centre[1] - 31)
                current_bounding_box: Tuple[int, int, int, int, str, str, float] = (
                    bounding_box_origin[0], bounding_box_origin[1], template.shape[0] - 1, template.shape[1] - 1, label_top, label_bottom, pred_value
                )
                bounding_boxes.add(current_bounding_box)
                bounding_box_removals: Set[Tuple[int, int, int, int, str, str, float]] = set()
                for bounding_box in bounding_boxes:
                    if bounding_box != current_bounding_box:
                        if abs(bounding_box[0] - current_bounding_box[0]) <= template.shape[0] and abs(bounding_box[1] - current_bounding_box[1]) <= template.shape[1]:
                            bounding_box_removals.add(bounding_box if current_bounding_box[6] < bounding_box[6] else current_bounding_box)
                bounding_boxes.difference_update(bounding_box_removals)
        progress.update(1)
    render_bounding_boxes(image_predict, bounding_boxes)
    write_annotation_file(predict_annotation_directory_path, image_name, bounding_boxes)
    cv.imwrite(f"{predict_image_directory_path}/{image_name}", image_predict)


def write_annotation_file(predict_annotation_directory_path: Path, image_name: str, bounding_boxes: Set[Tuple[int, int, int, int, str, str, float]]) -> None:
    """Writes the given bounding boxes to an annotation file."""
    annotation_lines = [f"{bounding_box[5]}, ({bounding_box[0]}, {bounding_box[1]}), ({bounding_box[0] + bounding_box[2]}, {bounding_box[1] + bounding_box[3]})\n" for bounding_box in bounding_boxes]
    with open(f"{predict_annotation_directory_path}/{Path(image_name).stem}.txt", "w") as annotation_file:
        annotation_file.writelines(annotation_lines)


def get_templates(train_directory_path: Path) -> Dict[str, Dict[int, np.ndarray]]: # Filename -> Sampling Level -> Template
    """Gets and crops training icons from the given directory."""
    icon_templates: Dict[str, Dict[int, np.ndarray]] = dict()
    for icon_path in train_directory_path.glob("*.png"):
        icon_bgra: np.ndarray = cv.imread(str(icon_path), cv.IMREAD_UNCHANGED)
        masked_icon: np.ndarray = mask_icon(icon_bgra)
        masked_icon_grayscale = cv.cvtColor(masked_icon, cv.COLOR_BGR2GRAY)
        icon_templates[icon_path.stem] = generate_gaussian_pyramid(masked_icon_grayscale, 6)
        # Crop icons
        # x, y, w, h = get_bounding_boxes(icon, 60000, 275000, 15)[0]
        # icon_crop: np.ndarray = icon[y:y + h, x:x + w]
        # cropped_icons[icon_path.stem] = icon_crop
    return icon_templates


def get_test_images(test_directory_path: Path) -> List[Tuple[str, np.ndarray]]:
    images: List[Tuple[str, np.ndarray]] = list()
    for image_path in test_directory_path.glob("*.png"):
        image: np.ndarray = cv.imread(str(image_path))
        image_grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        _, image_binary = cv.threshold(image_grayscale, 240, 255, cv.THRESH_BINARY)
        mask = cv.cvtColor(cv.bitwise_not(image_binary), cv.COLOR_GRAY2BGR)
        image_masked = cv.bitwise_and(image, mask)
        image_masked_grayscale = cv.cvtColor(image_masked, cv.COLOR_BGR2GRAY)
        images.append((image, image_masked_grayscale, image_path.name))
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
