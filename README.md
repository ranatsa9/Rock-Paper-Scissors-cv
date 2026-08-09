# Live Rock-Paper-Scissors Object Detection

An object-detection project that recognizes three hand-gesture classes:

- `paper`
- `rock`
- `scissors`

The repository contains the trained model, the Colab training notebook, and a Gradio application for live webcam and uploaded-image detection.

## Model results

The included model is the team's higher-accuracy three-class model. The live
camera uses 320-pixel inference for responsiveness, while uploaded images use
640-pixel inference for improved accuracy.

| Metric | Result |
| --- | ---: |
| mAP@50 | 96.0% |
| mAP@50-95 | 76.3% |
| Precision | 97.2% |
| Recall | 94.3% |

These reported validation results come from the training notebook associated
with the included model. Real webcam performance depends on lighting, camera
quality, background, distance, and whether those conditions appeared in the
training data.

## Repository structure

```text
.
|-- app.py
|-- dataset.yaml
|-- requirements.txt
|-- run_app.bat
|-- model/
|   `-- best.pt
`-- notebooks/
    `-- training_notebook.ipynb
```

## Run on Windows

1. Install Python 3.11 or 3.12 from [python.org](https://www.python.org/downloads/). During installation, enable **Add Python to PATH**.
2. Download or clone this repository.
3. Double-click `run_app.bat`.
4. The first launch creates a virtual environment and downloads dependencies, so it can take several minutes.
5. When the browser opens, allow camera access and select the **Live camera** tab.

Later launches reuse the installed environment and start faster.

## Run from a terminal

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

Open the local URL shown in the terminal, normally `http://127.0.0.1:7860`.

## Confidence threshold

- Lower the threshold if the model misses gestures.
- Raise it if the model produces incorrect detections.
- Live camera starts at `0.40` to reduce weak false detections.
- Uploaded images start at `0.25` to reduce missed gestures.

## Training and evaluation

The included runnable model is stored at `model/best.pt`. The notebook documents
the original four-class training workflow and is not required to run the app.

The combined training dataset is not committed because it contains thousands of images and is too large for a normal GitHub repository. Keep the dataset in shared cloud storage and document its source separately.

## Known limitation

No model is perfectly accurate outside its training conditions. If rock and
paper are still confused, retrain with balanced examples captured using the
same webcam, lighting, backgrounds, distances, and hand angles used in the app.
