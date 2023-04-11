import statistics
from pathlib import Path
from typing import Dict, List, Union

import cv2 as cv
import numpy as np


def main() -> None:
    pred_path_root: Path = Path("./predict/annotations")
    gt_path_root: Path = Path("./test/annotations")
    mean_ious = list()
    for pred_path, gt_path in zip(pred_path_root.rglob("*.txt"), gt_path_root.rglob("*.txt")):
        mean_ious.append(evaluate(pred_path, gt_path))
    print(statistics.mean(mean_ious))


def evaluate(path_pred: Union[str, Path], path_gt: Union[str, Path]) -> Dict[str, float]:
    results: Dict[str, float] = dict()
    with open(path_pred, "r", encoding="utf8") as pred_file:
        results_pred_raw: List[str] = sorted([line.strip() for line in pred_file.readlines()])
    with open(path_gt, "r", encoding="utf8") as gt_file:
        results_gt_raw: List[str] = sorted([line.strip() for line in gt_file.readlines()])
    results_raw = zip(results_pred_raw, results_gt_raw)
    ious = list()
    for result_pred_raw, result_gt_raw in results_raw:
        class_name_pred, *box_pred = parse_raw_result_string(result_pred_raw)
        class_name_gt, *box_gt = parse_raw_result_string(result_gt_raw)
        ious.append(calculate_iou(class_name_pred, box_pred, class_name_gt, box_gt))
    return statistics.mean(ious)


def parse_raw_result_string(result_string: str) -> List[float]:
    class_name, x1_raw, y1_raw, x2_raw, y2_raw = result_string.split(", ")
    x1, y1, x2, y2 = int(x1_raw.lstrip("(")), int(y1_raw.rstrip(")")), int(x2_raw.lstrip("(")), int(y2_raw.rstrip(")"))
    return class_name, x1, y1, x2, y2


def calculate_iou(class_name_pred: str, box_pred: List[float], class_name_gt: str, box_gt: List[float]) -> float:
    result = None
    if class_name_pred == class_name_gt:
        x1_pred, y1_pred, x2_pred, y2_pred = box_pred
        x1_gt, y1_gt, x2_gt, y2_gt = box_gt
        pred_area: int = (x2_pred - x1_pred) * (y2_pred - y1_pred)
        gt_area: int = (x2_gt - x1_gt) * (y2_gt - y1_gt)
        x1: int = max(x1_pred, x1_gt)
        y1: int = max(y1_pred, y1_gt)
        x2: int = min(x2_pred, x2_gt)
        y2: int = min(y2_pred, y2_gt)
        intersection_area: int = (x2 - x1) * (y2 - y1)
        union_area: int = pred_area + gt_area - intersection_area
        result = intersection_area / union_area
    else:
        result = 0.0
    return result


if __name__ == "__main__":
    main()