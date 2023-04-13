from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, List, Set, Tuple

import cv2 as cv
import numpy as np
from bounding_box import render_bounding_boxes
from method import SSD, ZMT
from template import generate_templates, ssd_match_template, zero_mean_match_template
from test_image import generate_test_images
from tqdm import tqdm

SSD_THRESHOLD: int = 165000
ZMT_THRESHOLD: int = 150000


def main() -> None:
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument("--test_directory_path", type=str, default="./test/images")
    parser.add_argument("--train_directory_path", type=str, default="./train/png")
    parser.add_argument("--method", type=str, default="ssd")
    parser.add_argument("--sampling_levels", type=int, default=5)
    args: Namespace = parser.parse_args()
    test_directory_path: Path = Path(args.test_directory_path).resolve(strict=True)
    train_directory_path: Path = Path(args.train_directory_path).resolve(strict=True)
    predict(test_directory_path, train_directory_path, args.method.lower(), args.sampling_levels)


def predict(
    test_directory_path: Path, train_directory_path: Path, method: str, sampling_levels: int
) -> None:
    """Predicts icon classes for all icons in all images in the given test directory."""
    predict_image_directory_root_path: str = "./predict/images"
    predict_annotation_directory_root_path: str = "./predict/annotations"
    for i in range(sampling_levels):
        predict_image_directory_path: Path = Path(f"{predict_image_directory_root_path}/{i}")
        predict_image_directory_path.mkdir(parents=True, exist_ok=True)
        predict_annotation_directory_path: Path = Path(
            f"{predict_annotation_directory_root_path}/{i}"
        )
        predict_annotation_directory_path.mkdir(parents=True, exist_ok=True)
    images: List[Tuple[np.ndarray, np.ndarray, str]] = generate_test_images(test_directory_path)
    templates: Dict[str, List[np.ndarray]] = generate_templates(
        train_directory_path, sampling_levels
    )
    progress: tqdm = tqdm(
        total=len(images) * len(templates) * sampling_levels, desc="Predicting Classes"
    )
    for image, image_masked, image_name in images:
        predict_icon_classes(
            method,
            sampling_levels,
            image_masked,
            image,
            image_name,
            templates,
            predict_image_directory_root_path,
            predict_annotation_directory_root_path,
            progress,
        )


def predict_icon_classes(
    method: str,
    n_sampling_levels: int,
    image_masked: np.ndarray,
    image_predict: np.ndarray,
    image_name: str,
    templates: Dict[str, List[np.ndarray]],
    predict_image_directory_root_path: str,
    predict_annotation_directory_root_path: str,
    progress: tqdm,
) -> None:
    """Predicts icon classes for all icons in the given image using the ssd method."""
    bounding_boxes: List[Set[Tuple[int, int, int, int, str, str, float]]] = [
        set() for _ in range(n_sampling_levels)
    ]
    for class_name, sampling_levels in templates.items():
        for sampling_level, template in enumerate(sampling_levels):
            # sampling_level = 3
            # template = sampling_levels[sampling_level]
            # for _ in range(1):
            if (
                template.shape[0] < image_masked.shape[0]
                and template.shape[1] < image_masked.shape[1]
            ):
                if method == SSD:
                    pred_value, pred_centre = ssd_match_template(image_masked, template)
                    if pred_value < SSD_THRESHOLD:
                        label_top, label_bottom = class_name.split("-", maxsplit=1)
                        bounding_box_origin: Tuple[int, int] = (
                            pred_centre[0] - ((template.shape[0] // 2) - 1),
                            pred_centre[1] - ((template.shape[1] // 2) - 1),
                        )
                        current_bounding_box: Tuple[int, int, int, int, str, str, float] = (
                            bounding_box_origin[0],
                            bounding_box_origin[1],
                            template.shape[0] - 1,
                            template.shape[1] - 1,
                            label_top,
                            label_bottom,
                            pred_value,
                        )
                        bounding_boxes[sampling_level].add(current_bounding_box)
                        bounding_box_removals: Set[
                            Tuple[int, int, int, int, str, str, float]
                        ] = set()
                        for bounding_box in bounding_boxes[sampling_level]:
                            if bounding_box != current_bounding_box:
                                if (
                                    abs(bounding_box[0] - current_bounding_box[0])
                                    <= template.shape[0]
                                    and abs(bounding_box[1] - current_bounding_box[1])
                                    <= template.shape[1]
                                ):
                                    bounding_box_removals.add(
                                        bounding_box
                                        if current_bounding_box[6] < bounding_box[6]
                                        else current_bounding_box
                                    )
                        bounding_boxes[sampling_level].difference_update(bounding_box_removals)
                if method == ZMT:
                    pred_value, pred_centre = zero_mean_match_template(image_masked, template)
                    if pred_value > ZMT_THRESHOLD:
                        label_top, label_bottom = class_name.split("-", maxsplit=1)
                        bounding_box_origin: Tuple[int, int] = (
                            pred_centre[0],
                            pred_centre[1],
                        )
                        current_bounding_box: Tuple[int, int, int, int, str, str, float] = (
                            bounding_box_origin[0],
                            bounding_box_origin[1],
                            template.shape[0] - 1,
                            template.shape[1] - 1,
                            label_top,
                            label_bottom,
                            pred_value,
                        )
                        bounding_boxes[sampling_level].add(current_bounding_box)
                        bounding_box_removals: Set[
                            Tuple[int, int, int, int, str, str, float]
                        ] = set()
                        for bounding_box in bounding_boxes[sampling_level]:
                            if bounding_box != current_bounding_box:
                                if (
                                    abs(bounding_box[0] - current_bounding_box[0])
                                    <= template.shape[0]
                                    and abs(bounding_box[1] - current_bounding_box[1])
                                    <= template.shape[1]
                                ):
                                    bounding_box_removals.add(
                                        bounding_box
                                        if current_bounding_box[6] > bounding_box[6]
                                        else current_bounding_box
                                    )
                        bounding_boxes[sampling_level].difference_update(bounding_box_removals)
                progress.update(1)
    for sampling_level, bounding_boxes_sampling_level in enumerate(bounding_boxes):
        image_predict_out: np.ndarray = np.copy(image_predict)
        render_bounding_boxes(image_predict_out, bounding_boxes_sampling_level)
        write_image_file(
            predict_image_directory_root_path, sampling_level, image_name, image_predict_out
        )
        write_annotation_file(
            predict_annotation_directory_root_path,
            sampling_level,
            image_name,
            bounding_boxes_sampling_level,
        )


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


if __name__ == "__main__":
    main()
