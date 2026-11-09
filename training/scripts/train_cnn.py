"""Train the local CNN U-Net model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from frontend.cnn_model import (
    YEARS,
    SolarCNNDataset,
    build_model,
    build_transforms,
    make_loader,
    train,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("training/data/COCO")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("training/results/runs/solar-seg/cnn-unet"),
    )
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--no-imagenet", action="store_true")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    dataset_root = args.dataset_root.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = SolarCNNDataset(
        dataset_root, YEARS, "train", build_transforms(train=True)
    )
    valid_dataset = SolarCNNDataset(
        dataset_root, YEARS, "valid", build_transforms(train=False)
    )
    train_loader = make_loader(
        train_dataset, args.batch_size, args.workers, True, device
    )
    valid_loader = make_loader(
        valid_dataset, args.batch_size, args.workers, False, device
    )
    model = build_model(None if args.no_imagenet else "imagenet").to(device)
    weights = train(
        model,
        train_loader,
        valid_loader,
        len(train_dataset),
        len(valid_dataset),
        args.epochs,
        args.output_dir,
        device,
    )
    print(f"Best weights: {weights}")


if __name__ == "__main__":
    main()
