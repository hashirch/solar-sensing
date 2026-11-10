# Solar Sensing

This repository contains a multi-temporal (2021–2025) solar-panel
segmentation dataset and CNN/U-Net and Mask R-CNN training workflows.

## Repository Structure

```text
training/
├── data/                         # Datasets used during training/evaluation
├── models/                       # Training-only model assets
├── notebooks/                    # CNN/U-Net and Mask R-CNN notebooks
├── results/                      # Training outputs and archives
└── scripts/                     # Training and evaluation scripts
frontend/
├── app.py                       # Legacy Streamlit inference frontend
├── models/                      # Checkpoints required by the frontend
└── src/                         # React/Vite workspace UI
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
python -m training.scripts.train_cnn
python -m training.scripts.evaluate_cnn

python -m training.scripts.train_mask_rcnn
python -m training.scripts.evaluate_mask_rcnn
```

The default evaluation checkpoints are:

- `frontend/models/best_cnn_unet.pth`
- `frontend/models/mask_rcnn_final.pth`

Use `--device cpu` when CUDA is unavailable. For a 6 GB GPU, use a small
Mask R-CNN batch size:

```bash
python -m training.scripts.train_mask_rcnn --batch-size 1 --workers 0
```

## Professional React Workspace UI

The primary frontend is a professional multi-screen React/Vite application featuring top navigation, interactive inspection modals, and responsive comparison views:

- **Executive Dashboard (`/`)**: System diagnostics, dual-model pipeline status, and quick module links.
- **Inference Workspace (`/workspace`)**: Dual-engine image segmentation with benchmark sample scenes, queue management, cumulative statistics (total panels, surface area in m², estimated kWp), and side-by-side comparison gallery.
- **Interactive Inspection Modal**: Features an interactive split-view comparison slider (raw vs predicted), side-by-side synchronized view, per-instance bounding box audit table, and CSV export.
- **Model Architecture & Benchmarks (`/models`)**: Technical deep-dive for CNN/U-Net and Detectron2 Mask R-CNN, including interactive confidence and NMS IoU threshold exploration sliders.
- **Research Methodology (`/research`)**: Multi-temporal (2021–2025) solar corpus documentation, training pipeline specifications, and evaluation artifacts.

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. The client runs high-fidelity client-side local inference in preview mode and keeps uploaded images on your machine. Set `VITE_API_URL` when connecting to a remote inference bridge.

## Legacy Streamlit Inference Frontend

Install the frontend dependencies into the system Python environment:

```bash
./install.sh
```

After installation, start the frontend with:

```bash
./run.sh
```

You can pass Streamlit options through to the script, for example:

```bash
./run.sh --server.port 8501
```

To start it manually instead:

```bash
streamlit run frontend/app.py
```

Open the displayed local URL, upload one or more images, and select CNN/U-Net,
Mask R-CNN, or both. Each result includes an expandable full-size view.

### Optional email OTP gate

The Streamlit app includes the email OTP flow adapted from the
`ovula-pcos-tracker-FYP-Project` authentication module. It is disabled by
default so local inference remains unchanged. Enable it with
`SOLARSIGHT_OTP_REQUIRED=true` and configure SMTP credentials:

```bash
export SOLARSIGHT_OTP_REQUIRED=true
export SMTP_USERNAME=solarsensing@gmail.com
export SMTP_PASSWORD='your-gmail-app-password'
export FROM_EMAIL=solarsensing@gmail.com
streamlit run frontend/app.py
```

For local development, copy `.env.example` to `.env` and replace
`SMTP_PASSWORD` with a Gmail App Password. `run.sh` loads `.env`
automatically, including App Passwords that contain spaces:

```bash
cp .env.example .env
./run.sh
```

The OTP service reads these values with `os.getenv`; it does not store
credentials in source code. Never commit `.env`.

Codes are six numeric digits, expire after five minutes by default,
and are compared using a constant-time check. Set `OTP_EXPIRY_MINUTES` to
change the expiry window. Never commit SMTP credentials.
