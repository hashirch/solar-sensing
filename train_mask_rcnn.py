"""
Mask R-CNN Training Script for Solar Panel Segmentation
========================================================
Uses Detectron2 with the COCO-format annotations.
Optimized for RTX 3050 (6GB VRAM).
"""

import json
import os
from pathlib import Path

import torch

# Detectron2 imports
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader


COCO_ROOT = Path("solar-panel-exports/COCO")
YEARS = ["2021", "2022", "2023", "2024", "2025"]


def register_datasets():
    """Register all year/split combinations as COCO datasets."""
    for year in YEARS:
        for split in ["train", "valid", "test"]:
            name = f"solar_{year}_{split}"
            img_dir = str(COCO_ROOT / year / split / "images")
            ann_file = str(COCO_ROOT / year / split / "_annotations.coco.json")

            if os.path.exists(ann_file):
                if name not in DatasetCatalog.list():
                    register_coco_instances(name, {}, ann_file, img_dir)

    # Also register combined datasets (all years merged)
    _register_combined("train")
    _register_combined("valid")
    _register_combined("test")


def _register_combined(split):
    """Merge all years for a given split into one dataset."""
    combined_name = f"solar_all_{split}"
    if combined_name in DatasetCatalog.list():
        return

    def loader():
        all_records = []
        img_id_offset = 0
        ann_id_offset = 0

        for year in YEARS:
            ann_file = COCO_ROOT / year / split / "_annotations.coco.json"
            img_dir = COCO_ROOT / year / split / "images"

            if not ann_file.exists():
                continue

            data = json.loads(ann_file.read_text())

            # Build image ID mapping for this year
            old_to_new_img = {}
            for img in data.get("images", []):
                old_id = img["id"]
                new_id = old_id + img_id_offset
                old_to_new_img[old_id] = new_id

                record = {
                    "file_name": str(img_dir / Path(img["file_name"]).name),
                    "image_id": new_id,
                    "height": img["height"],
                    "width": img["width"],
                    "annotations": [],
                }
                all_records.append(record)

            # Map annotations to records
            record_map = {r["image_id"]: r for r in all_records if r["image_id"] >= img_id_offset}
            for ann in data.get("annotations", []):
                new_img_id = old_to_new_img.get(ann["image_id"])
                if new_img_id is None or new_img_id not in record_map:
                    continue
                record_map[new_img_id]["annotations"].append({
                    "bbox": ann["bbox"],
                    "bbox_mode": 1,  # BoxMode.XYWH_ABS
                    "segmentation": ann["segmentation"],
                    "category_id": 0,  # Force single class
                    "iscrowd": ann.get("iscrowd", 0),
                })

            if data.get("images"):
                img_id_offset = max(old_to_new_img.values()) + 1
            if data.get("annotations"):
                ann_id_offset += len(data["annotations"])

        return all_records

    DatasetCatalog.register(combined_name, loader)
    MetadataCatalog.get(combined_name).set(thing_classes=["solar_panel"])


class SolarTrainer(DefaultTrainer):
    """Custom trainer with COCO evaluation."""

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "eval")
        return COCOEvaluator(dataset_name, output_dir=output_folder)


def main():
    register_datasets()

    cfg = get_cfg()

    # Use Mask R-CNN with ResNet-50 FPN backbone
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )

    # Dataset
    cfg.DATASETS.TRAIN = ("solar_all_train",)
    cfg.DATASETS.TEST = ("solar_all_valid",)

    # Solver — tuned for RTX 3050 (6GB)
    cfg.SOLVER.IMS_PER_BATCH = 2          # 2 images per batch (VRAM-safe)
    cfg.SOLVER.BASE_LR = 0.0025
    cfg.SOLVER.MAX_ITER = 5000
    cfg.SOLVER.STEPS = (3000, 4000)       # LR decay schedule
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000
    cfg.SOLVER.WARMUP_ITERS = 500

    # Model
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1   # solar_panel only
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Input
    cfg.INPUT.MIN_SIZE_TRAIN = (400, 432, 480)
    cfg.INPUT.MAX_SIZE_TRAIN = 512
    cfg.INPUT.MIN_SIZE_TEST = 432
    cfg.INPUT.MAX_SIZE_TEST = 512

    # Dataloader
    cfg.DATALOADER.NUM_WORKERS = 4

    # Output
    cfg.OUTPUT_DIR = "runs/solar-seg/mask-rcnn"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Save config for reproducibility
    with open(os.path.join(cfg.OUTPUT_DIR, "config.yaml"), "w") as f:
        f.write(cfg.dump())

    print("=" * 60)
    print("Starting Mask R-CNN training")
    print(f"  Backbone: ResNet-50 + FPN")
    print(f"  Max iterations: {cfg.SOLVER.MAX_ITER}")
    print(f"  Batch size: {cfg.SOLVER.IMS_PER_BATCH}")
    print(f"  Learning rate: {cfg.SOLVER.BASE_LR}")
    print(f"  Device: {cfg.MODEL.DEVICE}")
    print("=" * 60)

    # Train
    trainer = SolarTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    # Evaluate on test set
    print("\n" + "=" * 60)
    print("Evaluating on test set...")
    print("=" * 60)

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.DATASETS.TEST = ("solar_all_test",)
    predictor = DefaultPredictor(cfg)
    evaluator = COCOEvaluator("solar_all_test", output_dir=cfg.OUTPUT_DIR)
    test_loader = build_detection_test_loader(cfg, "solar_all_test")
    results = inference_on_dataset(predictor.model, test_loader, evaluator)

    print("\n" + "=" * 60)
    print("Test Set Results (Mask R-CNN):")
    if "segm" in results:
        print(f"  Segmentation mAP50:    {results['segm']['AP50']:.2f}")
        print(f"  Segmentation mAP50-95: {results['segm']['AP']:.2f}")
    if "bbox" in results:
        print(f"  BBox mAP50:            {results['bbox']['AP50']:.2f}")
        print(f"  BBox mAP50-95:         {results['bbox']['AP']:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
