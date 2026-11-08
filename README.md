# Solar Panel Segmentation

This repository contains a multi-temporal (2021–2025) solar-panel
segmentation dataset and CNN/U-Net and Mask R-CNN training workflows.

## Repository Structure

```text
data/
├── COCO/                         # Images and COCO segmentation annotations
└── download_datasets.py         # COCO dataset download and organization
frontend/
├── app.py                       # Legacy Streamlit inference frontend
└── src/                         # React/Vite workspace UI
models/                           # Additional model assets
notebooks/
├── cnn.ipynb                    # CNN/U-Net notebook
└── rcnn.ipynb                   # Mask R-CNN notebook
results/
├── weights/                     # Trained CNN and Mask R-CNN checkpoints
└── runs/                        # Generated training/evaluation outputs
scripts/
├── train_cnn.py
├── evaluate_cnn.py
├── train_mask_rcnn.py
└── evaluate_mask_rcnn.py
src/solar_segmentation/
└── cnn.py                       # Shared CNN dataset/model/training utilities
```

## Install

```bash
pip install -r requirements-local.txt
```

Detectron2 must be installed in a compatible PyTorch environment for Mask
R-CNN training, evaluation, and frontend use:

```bash
python -m pip install --no-build-isolation \
  'git+https://github.com/facebookresearch/detectron2.git'
```

## Separate Training and Evaluation

```bash
python -m scripts.train_cnn
python -m scripts.evaluate_cnn

python -m scripts.train_mask_rcnn
python -m scripts.evaluate_mask_rcnn
```

The default evaluation checkpoints are:

- `results/weights/best_cnn_unet.pth`
- `results/weights/mask_rcnn_final.pth`

Use `--device cpu` when CUDA is unavailable. For a 6 GB GPU, use a small
Mask R-CNN batch size:

```bash
python -m scripts.train_mask_rcnn --batch-size 1 --workers 0
```

## React Workspace UI

The primary frontend is a React/Vite workspace with responsive upload,
comparison, and result-preview flows:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. The client currently runs in preview mode and
keeps uploaded images local. Set `VITE_API_URL` when connecting it to an
inference API.

## Legacy Streamlit Inference Frontend

```bash
streamlit run frontend/app.py
```

Open the displayed local URL, upload one or more images, and select CNN/U-Net,
Mask R-CNN, or both. Each result includes an expandable full-size view.
