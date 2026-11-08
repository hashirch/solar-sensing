"""Evaluate a trained CNN U-Net model on the local test split."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.solar_segmentation.cnn import (
    YEARS,
    SolarCNNDataset,
    build_model,
    build_transforms,
    evaluate,
    make_loader,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/COCO"))
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("results/weights/best_cnn_unet.pth"),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    dataset = SolarCNNDataset(
        args.dataset_root.expanduser().resolve(),
        YEARS,
        "test",
        build_transforms(train=False),
    )
    loader = make_loader(dataset, args.batch_size, args.workers, False, device)
    model = build_model(None).to(device)
    model.load_state_dict(
        torch.load(args.weights.expanduser().resolve(), map_location=device, weights_only=True)
    )
    mean_iou, mean_f1 = evaluate(model, loader, device)
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Mean Dice/F1: {mean_f1:.4f}")


if __name__ == "__main__":
    main()
