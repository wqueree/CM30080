from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, List, Set, Tuple

import cv2 as cv
import numpy as np
from bounding_box import render_bounding_boxes
from template import generate_templates, zero_mean_match_template, ssd_match_template
from test_image import generate_test_images
from tqdm import tqdm


SSD_THRESHOLD: int = 165000
ZMT_THRESHOLD: int = 150000


def main() -> None:
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument("--test_directory_path", type=str, default="./test/images")
    parser.add_argument("--train_directory_path", type=str, default="./train/png")
    parser.add_argument("--method", type=str, default="ssd")
    args: Namespace = parser.parse_args()
    test_directory_path: Path = Path(args.test_directory_path).resolve(strict=True)
    train_directory_path: Path = Path(args.train_directory_path).resolve(strict=True)
    predict(test_directory_path, train_directory_path, args.method)


def predict(test_directory_path: Path, train_directory_path: Path, method: str) -> None:
    """Predicts icon classes for all icons in all images in the given test directory."""
    predict_image_directory_path: Path = Path("./predict/images")
    predict_image_directory_path.mkdir(parents=True, exist_ok=True)
    predict_annotation_directory_path: Path = Path("./predict/annotations")
    predict_annotation_directory_path.mkdir(parents=True, exist_ok=True)
    images: List[str, np.ndarray] = generate_test_images(test_directory_path)
    templates: Dict[str, Dict[int, np.ndarray]] = generate_templates(train_directory_path)
    progress: tqdm = tqdm(total=len(images) * len(templates), desc="Predicting Classes")
    for image, image_masked, image_name in images:
        predict_ssd_icon_classes(image_masked, image, image_name, templates, predict_image_directory_path, predict_annotation_directory_path, progress)


def predict_ssd_icon_classes(image_masked: np.ndarray, image_predict: np.ndarray, image_name: str, templates: Dict[str, np.ndarray], predict_image_directory_path: Path, predict_annotation_directory_path: Path, progress: tqdm) -> str:
    """Predicts icon classes for all icons in the given image using the ssd method."""
    bounding_boxes: Set[Tuple[int]] = set()
    for class_name, sampling_levels in templates.items():
        template = sampling_levels[3]
        if template.shape[0] < image_masked.shape[0] and template.shape[1] < image_masked.shape[1]:
            pred_value, pred_centre = ssd_match_template(image_masked, template)
            if pred_value < SSD_THRESHOLD:
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


def predict_zmt_icon_classes(image_masked: np.ndarray, image_predict: np.ndarray, image_name: str, templates: Dict[str, np.ndarray], predict_image_directory_path: Path, predict_annotation_directory_path: Path, progress: tqdm) -> str:
    """Predicts icon classes for all icons in the given image using the zero mean template method."""
    bounding_boxes: Set[Tuple[int]] = set()
    for class_name, sampling_levels in templates.items():
        template = sampling_levels[3]
        if template.shape[0] < image_masked.shape[0] and template.shape[1] < image_masked.shape[1]:
            pred_value, pred_centre = zero_mean_match_template(image_masked, template)
            print(f"{class_name}: {pred_value}")
            if pred_value > ZMT_THRESHOLD:
                label_top, label_bottom = class_name.split("-", maxsplit=1)
                bounding_box_origin: Tuple[int, int] = (pred_centre[0], pred_centre[1])
                current_bounding_box: Tuple[int, int, int, int, str, str, float] = (
                    bounding_box_origin[0], bounding_box_origin[1], template.shape[0] - 1, template.shape[1] - 1, label_top, label_bottom, pred_value
                )
                bounding_boxes.add(current_bounding_box)
                bounding_box_removals: Set[Tuple[int, int, int, int, str, str, float]] = set()
                for bounding_box in bounding_boxes:
                    if bounding_box != current_bounding_box:
                        if abs(bounding_box[0] - current_bounding_box[0]) <= template.shape[0] and abs(bounding_box[1] - current_bounding_box[1]) <= template.shape[1]:
                            bounding_box_removals.add(bounding_box if current_bounding_box[6] > bounding_box[6] else current_bounding_box)
                bounding_boxes.difference_update(bounding_box_removals)
        progress.update(1)
    render_bounding_boxes(image_predict, bounding_boxes)
    cv.imshow("image", image_predict)
    cv.waitKey(0)
    cv.destroyAllWindows()
    # write_annotation_file(predict_annotation_directory_path, image_name, bounding_boxes)
    # cv.imwrite(f"{predict_image_directory_path}/{image_name}", image_predict)


def write_annotation_file(predict_annotation_directory_path: Path, image_name: str, bounding_boxes: Set[Tuple[int, int, int, int, str, str, float]]) -> None:
    """Writes the given bounding boxes to an annotation file."""
    annotation_lines = [f"{bounding_box[5]}, ({bounding_box[0]}, {bounding_box[1]}), ({bounding_box[0] + bounding_box[2]}, {bounding_box[1] + bounding_box[3]})\n" for bounding_box in bounding_boxes]
    with open(f"{predict_annotation_directory_path}/{Path(image_name).stem}.txt", "w") as annotation_file:
        annotation_file.writelines(annotation_lines)


if __name__ == "__main__":
    main()
