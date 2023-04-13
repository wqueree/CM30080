import statistics
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Union


def main() -> None:
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument("--pred_path_root", type=str, default="./predict/annotations")
    parser.add_argument("--gt_path_root", type=str, default="./test/annotations")
    args: Namespace = parser.parse_args()
    pred_path_root: Path = Path(args.pred_path_root).resolve(strict=True)
    gt_path_root: Path = Path(args.gt_path_root).resolve(strict=True)
    results: Dict[str, float] = evaluate_results(pred_path_root, gt_path_root)
    print(results)


def evaluate_results(pred_path_root: Path, gt_path_root: Path) -> Dict[str, float]:
    """Evaluates the results of all results in the given prediction and ground truth paths."""
    results: Dict[str, float] = dict()
    raw_results: Dict[str, List[float]] = defaultdict(list)
    for pred_path, gt_path in zip(pred_path_root.rglob("*.txt"), gt_path_root.rglob("*.txt")):
        evaluate_result(pred_path, gt_path, raw_results)
    tp: float = sum(raw_results["tp"])
    fp: float = sum(raw_results["fp"])
    fn: float = sum(raw_results["fn"])
    results["iou"] = statistics.mean(raw_results["iou"]) if raw_results["iou"] else 0.0
    results["precision"] = tp / (tp + fp) if tp or fp else 0.0
    results["recall"] = tp / (tp + fn) if tp or fn else 0.0
    return results


def evaluate_result(
    path_pred: Union[str, Path], path_gt: Union[str, Path], raw_results: Dict[str, List[float]]
) -> None:
    """Evaluates the result of the given prediction and ground truth paths."""
    with open(path_pred, "r", encoding="utf8") as pred_file:
        results_pred_raw: List[str] = sorted([line.strip() for line in pred_file.readlines()])
    with open(path_gt, "r", encoding="utf8") as gt_file:
        results_gt_raw: List[str] = sorted([line.strip() for line in gt_file.readlines()])
    results_pred: Dict[str, Tuple[int, int, int, int]] = generate_results_dict(results_pred_raw)
    results_gt: Dict[str, Tuple[int, int, int, int]] = generate_results_dict(results_gt_raw)
    for class_name, box_pred in results_pred.items():
        if class_name in results_gt:
            iou: float = calculate_iou(box_pred, results_gt[class_name])
            raw_results["iou"].append(iou)
            raw_results["tp" if iou > 0.5 else "fp"].append(1.0)
        else:
            raw_results["fp"].append(1.0)
    for class_name, box_pred in results_gt.items():
        if class_name not in results_pred:
            raw_results["fn"].append(1.0)


def generate_results_dict(results_raw: List[str]) -> Dict[str, Tuple[int, int, int, int]]:
    """Generates a dictionary of results from the given raw results."""
    results: Dict[str, Tuple[int, int, int, int]] = dict()
    for result_raw in results_raw:
        class_name, box = parse_raw_result_string(result_raw)
        results[class_name] = box
    return results


def parse_raw_result_string(result_string: str) -> Tuple[str, Tuple[int, int, int, int]]:
    """Parses the given raw result string into a list of floats."""
    class_name, x1_raw, y1_raw, x2_raw, y2_raw = result_string.split(", ")
    box: Tuple[int, int, int, int] = (
        int(x1_raw.lstrip("(")),
        int(y1_raw.rstrip(")")),
        int(x2_raw.lstrip("(")),
        int(y2_raw.rstrip(")")),
    )
    return class_name, box


def calculate_iou(box_pred: Tuple[int, int, int, int], box_gt: Tuple[int, int, int, int]) -> float:
    """Calculates the intersection over union of the given prediction and ground truth boxes."""
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
    return intersection_area / union_area if union_area > 0 else 0.0


if __name__ == "__main__":
    main()
