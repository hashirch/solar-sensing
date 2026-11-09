# Solar Panel Segmentation

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
export SMTP_USERNAME=your-smtp-user
export SMTP_PASSWORD=your-smtp-password
export FROM_EMAIL=your-sender@example.com
streamlit run frontend/app.py
```

Codes are six hexadecimal characters, expire after five minutes by default,
and are compared using a constant-time check. Set `OTP_EXPIRY_MINUTES` to
change the expiry window. Never commit SMTP credentials.
