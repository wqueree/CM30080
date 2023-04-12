import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Union

import cv2 as cv
import numpy as np


def main() -> None:
    pred_path_root: Path = Path("./predict/annotations")
    gt_path_root: Path = Path("./test/annotations")
    results: Dict[str, float] = evaluate_results(pred_path_root, gt_path_root)
    print(results)


def evaluate_results(pred_path_root: Path, gt_path_root: Path) -> Dict[str, float]:
    """Evaluates the results of all results in the given prediction and ground truth paths."""
    results: Dict[str, float] = dict()
    raw_results: Dict[str, List[float]] = defaultdict(list)
    for pred_path, gt_path in zip(pred_path_root.rglob("*.txt"), gt_path_root.rglob("*.txt")):
        evaluate_result(pred_path, gt_path, raw_results)
    results["iou"] = statistics.mean(raw_results["iou"])
    return results


def evaluate_result(path_pred: Union[str, Path], path_gt: Union[str, Path], raw_results: Dict[str, List[float]]) -> None:
    """Evaluates the result of the given prediction and ground truth paths."""
    results: Dict[str, float] = dict()
    with open(path_pred, "r", encoding="utf8") as pred_file:
        results_pred_raw: List[str] = sorted([line.strip() for line in pred_file.readlines()])
    with open(path_gt, "r", encoding="utf8") as gt_file:
        results_gt_raw: List[str] = sorted([line.strip() for line in gt_file.readlines()])
    results_raw = zip(results_pred_raw, results_gt_raw)
    for result_pred_raw, result_gt_raw in results_raw:
        class_name_pred, *box_pred = parse_raw_result_string(result_pred_raw)
        class_name_gt, *box_gt = parse_raw_result_string(result_gt_raw)
        raw_results["iou"].append(calculate_iou(class_name_pred, box_pred, class_name_gt, box_gt))


def parse_raw_result_string(result_string: str) -> Tuple[str, int, int, int, int]:
    """Parses the given raw result string into a list of floats."""
    class_name, x1_raw, y1_raw, x2_raw, y2_raw = result_string.split(", ")
    x1, y1, x2, y2 = int(x1_raw.lstrip("(")), int(y1_raw.rstrip(")")), int(x2_raw.lstrip("(")), int(y2_raw.rstrip(")"))
    return class_name, x1, y1, x2, y2


def calculate_iou(class_name_pred: str, box_pred: List[int], class_name_gt: str, box_gt: List[int]) -> float:
    """Calculates the intersection over union of the given prediction and ground truth boxes."""
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
        return intersection_area / union_area
    return 0.0


if __name__ == "__main__":
    main()