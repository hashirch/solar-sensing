import json
import os
import re
import shutil
from pathlib import Path


WORKSPACE = "hashir-auqmy"
MODEL = "solar-panel-segmentation-sj9ak"
VERSION = 1
YEARS = {"2021", "2022", "2023", "2024", "2025"}

root = Path("solar-panel-exports")
staging = root / "_downloads"
coco_out = root / "COCO"
yolo_out = root / "YOLO"

import subprocess

GITHUB_REPO = "https://github.com/Hinarashid220/FYP.git"

# If ROBOFLOW_API_KEY is not set, download directly from GitHub repo
api_key = os.environ.get("ROBOFLOW_API_KEY")
if not api_key:
    print(f"ROBOFLOW_API_KEY not found. Fetching dataset directly from GitHub: {GITHUB_REPO}")
    dest = Path("FYP")
    if not dest.exists() and not Path("Dataset/COCO").exists():
        subprocess.run(["git", "clone", GITHUB_REPO], check=True)
        print("Cloned repository successfully into ./FYP")
    else:
        print("Dataset already available locally.")
    exit(0)

    from roboflow import Roboflow
    rf = Roboflow(api_key=api_key)
version = rf.workspace(WORKSPACE).project(MODEL).version(VERSION)

# Download both segmentation formats.
coco_ds = version.download(
    model_format="coco-segmentation",
    location=str(staging / "coco"),
    overwrite=True,
)
yolo_ds = version.download(
    model_format="yolov8",
    location=str(staging / "yolo"),
    overwrite=True,
)

def get_year(filename):
    match = re.search(r"(2021|2022|2023|2024|2025)", filename)
    if not match:
        raise ValueError(f"Could not determine year from: {filename}")
    return match.group(1)

# Rebuild COCO as:
# COCO/2021/train/images + _annotations.coco.json
# COCO/2021/valid/images + _annotations.coco.json
# ...
for annotation_file in Path(coco_ds.location).rglob("_annotations.coco.json"):
    split = annotation_file.parent.name
    if split not in {"train", "valid", "test"}:
        split = "train"

    data = json.loads(annotation_file.read_text())
    annotations_by_image = {}
    for annotation in data.get("annotations", []):
        annotations_by_image.setdefault(annotation["image_id"], []).append(annotation)

    grouped = {}
    for image in data.get("images", []):
        year = get_year(image["file_name"])
        grouped.setdefault(year, {"images": [], "annotations": []})
        grouped[year]["images"].append(image)
        grouped[year]["annotations"].extend(
            annotations_by_image.get(image["id"], [])
        )

        source = annotation_file.parent / image["file_name"]
        destination = coco_out / year / split / "images" / Path(
            image["file_name"]
        ).name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    for year, records in grouped.items():
        target = coco_out / year / split
        target.mkdir(parents=True, exist_ok=True)

        output = {
            "info": data.get("info", {}),
            "licenses": data.get("licenses", []),
            "categories": data.get("categories", []),
            "images": [
                {**image, "file_name": f"images/{Path(image['file_name']).name}"}
                for image in records["images"]
            ],
            "annotations": records["annotations"],
        }
        (target / "_annotations.coco.json").write_text(
            json.dumps(output, indent=2)
        )

# Rebuild YOLO as:
# YOLO/2021/train/images + labels
# YOLO/2021/valid/images + labels
# ...
image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

for image in Path(yolo_ds.location).rglob("*"):
    if not image.is_file() or image.suffix.lower() not in image_extensions:
        continue
    if image.parent.name != "images":
        continue

    split = image.parent.parent.name
    year = get_year(image.name)

    image_target = yolo_out / year / split / "images" / image.name
    label_source = image.parent.parent / "labels" / f"{image.stem}.txt"
    label_target = yolo_out / year / split / "labels" / f"{image.stem}.txt"

    image_target.parent.mkdir(parents=True, exist_ok=True)
    label_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image, image_target)

    # Negative images correctly receive an empty label file.
    if label_source.exists():
        shutil.copy2(label_source, label_target)
    else:
        label_target.touch()

# Dataset configuration compatible with the rebuilt YOLO structure.
def yaml_list(split):
    return "\n".join(
        f"  - {year}/{split}/images" for year in sorted(YEARS)
    )

(yolo_out / "data.yaml").write_text(
    "path: .\n"
    "train:\n" + yaml_list("train") + "\n"
    "val:\n" + yaml_list("valid") + "\n"
    "test:\n" + yaml_list("test") + "\n"
    "names:\n"
    "  0: solar_panel\n"
)

# Produce the final ZIP archives.
shutil.make_archive(str(root / "solar-panels-COCO"), "zip", coco_out)
shutil.make_archive(str(root / "solar-panels-YOLO"), "zip", yolo_out)

print("Created:")
print(root / "solar-panels-COCO.zip")
print(root / "solar-panels-YOLO.zip")
