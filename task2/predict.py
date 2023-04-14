from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import cv2 as cv
import numpy as np
from bounding_box import render_bounding_boxes
from correlation import ssd_correlation
from template import generate_templates
from test_image import generate_test_images
from tqdm import tqdm

SSD_THRESHOLDS: Dict[int, int] = {
    0: 10000000,
    1: 5000000,
    2: 1500000,
    3: 165000,
    4: 100000,
}
BOUNDING_BOX_SIZE: int = 64


def predict(
    test_directory_path: Path, train_directory_path: Path, sampling_levels: List[int]
) -> None:
    """Predicts icon classes for all icons in all images in the given test directory."""
    predict_image_directory_root_path: str = "./predict/images"
    predict_annotation_directory_root_path: str = "./predict/annotations"
    for sampling_level in sampling_levels:
        predict_image_directory_path: Path = Path(
            f"{predict_image_directory_root_path}/{sampling_level}"
        )
        predict_image_directory_path.mkdir(parents=True, exist_ok=True)
        predict_annotation_directory_path: Path = Path(
            f"{predict_annotation_directory_root_path}/{sampling_level}"
        )
        predict_annotation_directory_path.mkdir(parents=True, exist_ok=True)
    images: List[Tuple[np.ndarray, np.ndarray, str]] = generate_test_images(test_directory_path)
    templates: Dict[str, Dict[int, np.ndarray]] = generate_templates(
        train_directory_path, sampling_levels
    )
    progress: tqdm = tqdm(
        total=len(images) * len(templates) * len(sampling_levels),
        desc="Predicting",
    )
    for image, image_masked, image_name in images:
        predict_icon_classes(
            image_masked,
            image,
            image_name,
            templates,
            predict_image_directory_root_path,
            predict_annotation_directory_root_path,
            progress,
        )


def predict_icon_classes(
    image_masked: np.ndarray,
    image_predict: np.ndarray,
    image_name: str,
    templates: Dict[str, Dict[int, np.ndarray]],
    predict_image_directory_root_path: str,
    predict_annotation_directory_root_path: str,
    progress: tqdm,
) -> None:
    """Predicts icon classes for all icons in the given image using the ssd method."""
    bounding_boxes: Dict[int, Set[Tuple[int, int, int, int, str, str, float]]] = defaultdict(set)
    for class_name, template_sampling_levels in templates.items():
        for template_sampling_level, template in template_sampling_levels.items():
            if (
                template.shape[0] < image_masked.shape[0]
                and template.shape[1] < image_masked.shape[1]
            ):
                pred_value, pred_centre = ssd_correlation(image_masked, template)
                if pred_value < SSD_THRESHOLDS[template_sampling_level]:
                    non_maximum_suppression(
                        class_name, pred_centre, pred_value, template_sampling_level, bounding_boxes
                    )
            progress.update(1)

    for template_sampling_level, bounding_boxes_sampling_level in bounding_boxes.items():
        image_predict_out: np.ndarray = np.copy(image_predict)
        render_bounding_boxes(image_predict_out, bounding_boxes_sampling_level)
        write_image_file(
            predict_image_directory_root_path,
            template_sampling_level,
            image_name,
            image_predict_out,
        )
        write_annotation_file(
            predict_annotation_directory_root_path,
            template_sampling_level,
            image_name,
            bounding_boxes_sampling_level,
        )


def non_maximum_suppression(
    class_name: str,
    pred_centre: Tuple[int, int],
    pred_value: float,
    template_sampling_level,
    bounding_boxes: Dict[int, Set[Tuple[int, int, int, int, str, str, float]]],
) -> None:
    """Breaks ties for overlapping boxes."""
    label_top, label_bottom = class_name.split("-", maxsplit=1)
    bounding_box_origin: Tuple[int, int] = (
        pred_centre[0] - ((BOUNDING_BOX_SIZE // 2) - 1),
        pred_centre[1] - ((BOUNDING_BOX_SIZE // 2) - 1),
    )
    current_bounding_box: Tuple[int, int, int, int, str, str, float] = (
        bounding_box_origin[0],
        bounding_box_origin[1],
        BOUNDING_BOX_SIZE,
        BOUNDING_BOX_SIZE,
        label_top,
        label_bottom,
        pred_value,
    )
    bounding_boxes[template_sampling_level].add(current_bounding_box)
    bounding_box_removals: Set[Tuple[int, int, int, int, str, str, float]] = set()
    for bounding_box in bounding_boxes[template_sampling_level]:
        if bounding_box != current_bounding_box:
            if (
                abs(bounding_box[0] - current_bounding_box[0]) <= BOUNDING_BOX_SIZE
                and abs(bounding_box[1] - current_bounding_box[1]) <= BOUNDING_BOX_SIZE
            ):
                bounding_box_removals.add(
                    bounding_box
                    if current_bounding_box[6] < bounding_box[6]
                    else current_bounding_box
                )
    bounding_boxes[template_sampling_level].difference_update(bounding_box_removals)


def write_image_file(
    predict_image_directory_path: str, sampling_level: int, image_name: str, image: np.ndarray
) -> None:
    """Writes the given image to a file."""
    cv.imwrite(f"{predict_image_directory_path}/{sampling_level}/{image_name}", image)


def write_annotation_file(
    predict_annotation_directory_path: str,
    sampling_level: int,
    image_name: str,
    bounding_boxes: Set[Tuple[int, int, int, int, str, str, float]],
) -> None:
    """Writes the given bounding boxes to an annotation file."""
    annotation_lines = [
        f"{bounding_box[5]}, ({bounding_box[0]}, {bounding_box[1]}), ({bounding_box[0] + bounding_box[2]}, {bounding_box[1] + bounding_box[3]})\n"
        for bounding_box in bounding_boxes
    ]
    with open(
        f"{predict_annotation_directory_path}/{sampling_level}/{Path(image_name).stem}.txt", "w"
    ) as annotation_file:
        annotation_file.writelines(annotation_lines)
