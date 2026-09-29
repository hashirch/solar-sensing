"""
Model Comparison & Temporal Analysis Script
=============================================
Run after both models are trained. Compares YOLO11-seg vs Mask R-CNN
and generates temporal growth analysis across 2021-2025.
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def compare_metrics():
    """Compare YOLO vs Mask R-CNN test metrics side by side."""
    print("=" * 70)
    print("MODEL COMPARISON: YOLO11m-seg vs Mask R-CNN (ResNet-50 FPN)")
    print("=" * 70)

    # Load YOLO results (if available)
    yolo_results = Path("runs/solar-seg/yolo11m-seg/results.csv")
    if yolo_results.exists():
        import csv
        with open(yolo_results) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if rows:
                last = rows[-1]
                print("\nYOLO11m-seg (best epoch):")
                for key, val in last.items():
                    key = key.strip()
                    if "map" in key.lower() or "loss" in key.lower():
                        print(f"  {key}: {float(val.strip()):.4f}")
    else:
        print("\n[!] YOLO results not found. Train first with: python train_yolo.py")

    # Load Mask R-CNN results (if available)
    maskrcnn_results = Path("runs/solar-seg/mask-rcnn/metrics.json")
    if maskrcnn_results.exists():
        data = json.loads(maskrcnn_results.read_text())
        print("\nMask R-CNN (final):")
        for key, val in data.items():
            if "segm" in key or "bbox" in key:
                print(f"  {key}: {val:.4f}")
    else:
        print("\n[!] Mask R-CNN results not found. Train first with: python train_mask_rcnn.py")


def temporal_analysis_yolo():
    """
    Run YOLO inference on all years and generate growth charts.
    Requires trained YOLO model.
    """
    from ultralytics import YOLO

    model_path = Path("runs/solar-seg/yolo11m-seg/weights/best.pt")
    if not model_path.exists():
        print("[!] YOLO model not found. Train first.")
        return

    model = YOLO(str(model_path))

    years = ["2021", "2022", "2023", "2024", "2025"]
    yolo_root = Path("solar-panel-exports/YOLO")
    output_dir = Path("runs/solar-seg/temporal-analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect statistics per year
    yearly_stats = {}
    base_stats = defaultdict(lambda: defaultdict(dict))  # base -> year -> stats

    for year in years:
        panel_counts = []
        total_area = 0
        images_with_panels = 0
        total_images = 0

        # Run on all splits
        for split in ["train", "valid", "test"]:
            img_dir = yolo_root / year / split / "images"
            if not img_dir.exists():
                continue

            results = model.predict(
                source=str(img_dir),
                imgsz=432,
                conf=0.25,
                save=False,
                verbose=False,
            )

            for r in results:
                total_images += 1
                n_panels = len(r.masks.data) if r.masks is not None else 0
                panel_counts.append(n_panels)

                if n_panels > 0:
                    images_with_panels += 1
                    area = r.masks.data.sum().item()
                    total_area += area

                    # Extract base name for per-location tracking
                    fname = Path(r.path).stem
                    # patch3_Base3_2025_jpg.rf.xxx -> Base3
                    parts = fname.split("_")
                    for p in parts:
                        if p.startswith("Base"):
                            base_name = p
                            patch_name = parts[0]
                            base_stats[base_name][year][patch_name] = n_panels
                            break

        yearly_stats[year] = {
            "total_images": total_images,
            "images_with_panels": images_with_panels,
            "total_panels_detected": sum(panel_counts),
            "avg_panels_per_image": np.mean(panel_counts) if panel_counts else 0,
            "max_panels_in_image": max(panel_counts) if panel_counts else 0,
            "total_pixel_area": total_area,
            "adoption_rate": images_with_panels / total_images if total_images else 0,
        }

        print(f"\n{year}:")
        for k, v in yearly_stats[year].items():
            if isinstance(v, float):
                print(f"  {k}: {v:.2f}")
            else:
                print(f"  {k}: {v}")

    # --- Generate Charts ---

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Solar Panel Growth Analysis (2021-2025)", fontsize=16, fontweight="bold")

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0"]

    # Chart 1: Total panels detected per year
    ax = axes[0, 0]
    counts = [yearly_stats[y]["total_panels_detected"] for y in years]
    bars = ax.bar(years, counts, color=colors, edgecolor="white", linewidth=1.5)
    ax.set_title("Total Solar Panels Detected", fontweight="bold")
    ax.set_ylabel("Panel Count")
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                str(c), ha="center", va="bottom", fontweight="bold")

    # Chart 2: Adoption rate (% of images with panels)
    ax = axes[0, 1]
    rates = [yearly_stats[y]["adoption_rate"] * 100 for y in years]
    ax.plot(years, rates, "o-", color="#E91E63", linewidth=2.5, markersize=10)
    ax.fill_between(years, rates, alpha=0.15, color="#E91E63")
    ax.set_title("Solar Adoption Rate", fontweight="bold")
    ax.set_ylabel("% of images with panels")
    ax.set_ylim(0, 100)

    # Chart 3: Average panels per image
    ax = axes[1, 0]
    avgs = [yearly_stats[y]["avg_panels_per_image"] for y in years]
    ax.plot(years, avgs, "s-", color="#4CAF50", linewidth=2.5, markersize=10)
    ax.fill_between(years, avgs, alpha=0.15, color="#4CAF50")
    ax.set_title("Avg Panels per Image", fontweight="bold")
    ax.set_ylabel("Average Count")

    # Chart 4: Heatmap of panels per base location over time
    ax = axes[1, 1]
    base_names = sorted(base_stats.keys(), key=lambda x: int(x.replace("Base", "")))
    heatmap_data = []
    for base in base_names:
        row = []
        for year in years:
            total = sum(base_stats[base].get(year, {}).values())
            row.append(total)
        heatmap_data.append(row)

    im = ax.imshow(heatmap_data, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(years)))
    ax.set_xticklabels(years)
    ax.set_yticks(range(len(base_names)))
    ax.set_yticklabels(base_names, fontsize=8)
    ax.set_title("Panels per Base Location", fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    chart_path = output_dir / "solar_growth_analysis.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nChart saved: {chart_path}")

    # Save raw stats as JSON
    stats_path = output_dir / "yearly_stats.json"
    stats_path.write_text(json.dumps(yearly_stats, indent=2))
    print(f"Stats saved: {stats_path}")


if __name__ == "__main__":
    compare_metrics()
    print("\n" + "=" * 70)
    print("TEMPORAL GROWTH ANALYSIS")
    print("=" * 70)
    temporal_analysis_yolo()
