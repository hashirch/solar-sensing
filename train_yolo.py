"""
YOLO11m-seg Training Script for Solar Panel Segmentation
=========================================================
Uses the pre-formatted YOLO dataset with data.yaml.
Optimized for RTX 3050 (6GB VRAM).
"""

from ultralytics import YOLO


def main():
    # Load YOLO11 medium segmentation model (pretrained on COCO)
    model = YOLO("yolo11m-seg.pt")

    # Train on solar panel dataset
    results = model.train(
        data="solar-panel-exports/YOLO/data.yaml",
        epochs=150,
        imgsz=432,
        batch=8,              # conservative for 6GB VRAM
        patience=30,          # early stopping if no improvement for 30 epochs
        workers=4,
        device=0,             # GPU 0
        # --- Augmentation (satellite-friendly) ---
        augment=True,
        mosaic=1.0,
        mixup=0.1,
        degrees=180,          # heavy rotation — satellite has no fixed orientation
        flipud=0.5,           # vertical flip
        fliplr=0.5,           # horizontal flip
        scale=0.3,            # ±30% scale
        hsv_h=0.015,          # slight hue variation
        hsv_s=0.5,            # saturation variation (lighting conditions)
        hsv_v=0.3,            # brightness variation
        # --- Output ---
        project="runs/solar-seg",
        name="yolo11m-seg",
        save=True,
        save_period=25,       # checkpoint every 25 epochs
        plots=True,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best model: runs/solar-seg/yolo11m-seg/weights/best.pt")
    print("=" * 60)

    # Validate on the test set
    model = YOLO("runs/solar-seg/yolo11m-seg/weights/best.pt")
    metrics = model.val(
        data="solar-panel-exports/YOLO/data.yaml",
        split="test",
        imgsz=432,
        batch=8,
        plots=True,
        project="runs/solar-seg",
        name="yolo11m-seg-test",
    )

    print("\n" + "=" * 60)
    print("Test Set Results:")
    print(f"  mAP50:    {metrics.seg.map50:.4f}")
    print(f"  mAP50-95: {metrics.seg.map:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
