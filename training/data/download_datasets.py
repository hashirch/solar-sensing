import json
import os
import re
import shutil
from pathlib import Path

from roboflow import Roboflow


WORKSPACE = "hashir-auqmy"
MODEL = "solar-panel-segmentation-sj9ak"
VERSION = 1
YEARS = {"2021", "2022", "2023", "2024", "2025"}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
root = PROJECT_ROOT / "data" / "exports"
staging = root / "_downloads"
coco_out = root / "COCO"

rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
version = rf.workspace(WORKSPACE).project(MODEL).version(VERSION)
coco_ds = version.download(
    model_format="coco-segmentation",
    location=str(staging / "coco"),
    overwrite=True,
)


def get_year(filename: str) -> str:
    match = re.search(r"(2021|2022|2023|2024|2025)", filename)
    if not match:
        raise ValueError(f"Could not determine year from: {filename}")
    return match.group(1)


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

shutil.make_archive(str(root / "solar-panels-COCO"), "zip", coco_out)
print("Created:", root / "solar-panels-COCO.zip")
