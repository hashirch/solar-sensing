"""Evaluate Detectron2 Mask R-CNN weights on the local test split."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

from training.scripts.train_mask_rcnn import YEARS, SPLITS, register_datasets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("training/data/COCO")
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("frontend/models/mask_rcnn_final.pth"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("training/results/runs/solar-seg/mask-rcnn-eval"),
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    register_datasets(args.dataset_root.expanduser().resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    )
    cfg.MODEL.WEIGHTS = str(args.weights.expanduser().resolve())
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.DEVICE = device
    cfg.INPUT.MIN_SIZE_TEST = 432
    cfg.INPUT.MAX_SIZE_TEST = 512
    cfg.DATASETS.TEST = ("solar_all_test",)
    predictor = DefaultPredictor(cfg)
    evaluator = COCOEvaluator("solar_all_test", output_dir=str(args.output_dir))
    loader = build_detection_test_loader(cfg, "solar_all_test")
    results = inference_on_dataset(predictor.model, loader, evaluator)
    for task in ("segm", "bbox"):
        if task in results:
            print(f"{task} AP50: {results[task]['AP50']:.2f}")
            print(f"{task} AP50-95: {results[task]['AP']:.2f}")


if __name__ == "__main__":
    main()
