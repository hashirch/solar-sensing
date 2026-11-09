"""Train the Detectron2 Mask R-CNN model locally."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator


YEARS = ("2021", "2022", "2023", "2024", "2025")
SPLITS = ("train", "valid", "test")


def register_datasets(root: Path) -> None:
    for year in YEARS:
        for split in SPLITS:
            name = f"solar_{year}_{split}"
            annotation = root / year / split / "_annotations.coco.json"
            images = root / year / split / "images"
            if annotation.exists() and name not in DatasetCatalog.list():
                register_coco_instances(name, {}, str(annotation), str(images))

    for split in SPLITS:
        name = f"solar_all_{split}"
        if name in DatasetCatalog.list():
            continue

        def loader(split: str = split) -> list[dict]:
            records: list[dict] = []
            offset = 0
            for year in YEARS:
                annotation = root / year / split / "_annotations.coco.json"
                if not annotation.exists():
                    continue
                data = json.loads(annotation.read_text())
                id_map = {}
                record_map = {}
                for image in data.get("images", []):
                    new_id = image["id"] + offset
                    id_map[image["id"]] = new_id
                    record = {
                        "file_name": str(
                            root / year / split / "images" / Path(image["file_name"]).name
                        ),
                        "image_id": new_id,
                        "height": image["height"],
                        "width": image["width"],
                        "annotations": [],
                    }
                    records.append(record)
                    record_map[new_id] = record
                for annotation_record in data.get("annotations", []):
                    new_id = id_map.get(annotation_record["image_id"])
                    if new_id not in record_map:
                        continue
                    record_map[new_id]["annotations"].append(
                        {
                            "bbox": annotation_record["bbox"],
                            "bbox_mode": 1,
                            "segmentation": annotation_record["segmentation"],
                            "category_id": 0,
                            "iscrowd": annotation_record.get("iscrowd", 0),
                        }
                    )
                if id_map:
                    offset = max(id_map.values()) + 1
            return records

        DatasetCatalog.register(name, loader)
        MetadataCatalog.get(name).set(thing_classes=["solar_panel"])


class SolarTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return COCOEvaluator(
            dataset_name,
            output_dir=output_folder or str(Path(cfg.OUTPUT_DIR) / "eval"),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("training/data/COCO")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("training/results/runs/solar-seg/mask-rcnn"),
    )
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    register_datasets(args.dataset_root.expanduser().resolve())
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    )
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
    cfg.DATASETS.TRAIN = ("solar_all_train",)
    cfg.DATASETS.TEST = ("solar_all_valid",)
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.SOLVER.BASE_LR = 0.0025
    cfg.SOLVER.MAX_ITER = args.max_iter
    cfg.SOLVER.STEPS = (int(args.max_iter * 0.6), int(args.max_iter * 0.8))
    cfg.SOLVER.WARMUP_ITERS = min(500, max(1, args.max_iter // 10))
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.DEVICE = device
    cfg.DATALOADER.NUM_WORKERS = args.workers
    cfg.OUTPUT_DIR = str(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer = SolarTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == "__main__":
    main()
