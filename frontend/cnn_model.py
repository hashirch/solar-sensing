"""Shared CNN U-Net model, dataset, and training helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import albumentations as A
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset


YEARS = ("2021", "2022", "2023", "2024", "2025")


class SolarCNNDataset(Dataset):
    def __init__(
        self,
        root: Path,
        years: Iterable[str],
        split: str,
        transform: A.Compose | None = None,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.samples: list[tuple[Path, np.ndarray]] = []
        for year in years:
            split_root = self.root / year / split
            annotation_path = split_root / "_annotations.coco.json"
            if not annotation_path.exists():
                continue
            data = json.loads(annotation_path.read_text())
            annotations: dict[int, list[dict]] = {}
            for annotation in data.get("annotations", []):
                annotations.setdefault(annotation["image_id"], []).append(annotation)
            for image in data.get("images", []):
                mask = np.zeros(
                    (image["height"], image["width"]), dtype=np.uint8
                )
                for annotation in annotations.get(image["id"], []):
                    polygons = annotation.get("segmentation", [])
                    for polygon in polygons:
                        points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
                        cv2.fillPoly(mask, [np.round(points).astype(np.int32)], 1)
                image_path = split_root / image["file_name"]
                self.samples.append((image_path, mask))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, mask = self.samples[index]
        image_data = cv2.imread(str(image_path))
        image = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB) if image_data is not None else None
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        return image, torch.as_tensor(mask, dtype=torch.float32).unsqueeze(0)


def build_transforms(train: bool) -> A.Compose:
    transforms: list[A.BasicTransform] = [
        A.Resize(512, 512),
    ]
    if train:
        transforms.extend(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
            ]
        )
    transforms.extend(
        [
            A.Normalize(),
            ToTensorV2(),
        ]
    )
    return A.Compose(transforms)


def build_model(encoder_weights: str | None) -> torch.nn.Module:
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=1,
        activation=None,
    )


def make_loader(
    dataset: Dataset,
    batch_size: int,
    workers: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )


def evaluate(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> tuple[float, float]:
    model.eval()
    intersection = union = predicted = target = 0.0
    with torch.no_grad():
        for images, masks in loader:
            logits = model(images.to(device))
            predictions = torch.sigmoid(logits) > 0.5
            targets = masks.to(device) > 0.5
            intersection += (predictions & targets).sum().item()
            union += (predictions | targets).sum().item()
            predicted += predictions.sum().item()
            target += targets.sum().item()
    iou = intersection / union if union else 1.0
    f1_denominator = predicted + target
    f1 = 2 * intersection / f1_denominator if f1_denominator else 1.0
    return iou, f1


def train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    train_size: int,
    valid_size: int,
    epochs: int,
    output_dir: Path,
    device: torch.device,
) -> Path:
    del train_size, valid_size
    criterion = smp.losses.DiceLoss(mode="binary", from_logits=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
    best_iou = -1.0
    best_path = output_dir / "best_cnn_unet.pth"
    for epoch in range(epochs):
        model.train()
        for images, masks in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(images.to(device)), masks.to(device))
            loss.backward()
            optimizer.step()
        scheduler.step()
        iou, _ = evaluate(model, valid_loader, device)
        print(f"Epoch {epoch + 1}/{epochs} - validation IoU: {iou:.4f}")
        if iou > best_iou:
            best_iou = iou
            torch.save(model.state_dict(), best_path)
    return best_path
