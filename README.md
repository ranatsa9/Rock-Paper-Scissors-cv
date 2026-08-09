# RPS Vision Arena — Streamlit Edition

A professional glassmorphism Streamlit interface for live rock-paper-scissors
detection using the team's three-class YOLO model. It includes three large
modes: continuous live detection, a camera snapshot, and an uploaded image.
Live mode also includes an optional two-player battle mode: the left hand is
Player 1, the right hand is Player 2, and the winner appears on the video.
The live camera is mirrored by default and uses 416-pixel inference for a
better accuracy/speed balance on Apple Silicon.
Fullscreen mode preserves the complete video frame and keeps game results
above the browser's video controls.

## Deploy on Streamlit Community Cloud

Connect this GitHub repository at https://share.streamlit.io and select
`app.py` as the entrypoint. The app includes a STUN configuration for remote
WebRTC camera connections.

## Run on macOS

### Terminal method

Open Terminal, change into this folder, and run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Later launches only need:

```bash
source .venv/bin/activate
python -m streamlit run app.py
```

Open http://localhost:8501 if the browser does not open automatically.

### Double-click method

Double-click `run_mac.command`. If macOS blocks it, right-click it, choose
**Open**, and confirm. The first launch installs dependencies and takes longer.

After the first setup, double-click `start_class.command` for a fast launch
without reinstalling packages. Prepare the `.venv` before class because it is
correctly excluded from Git and will not be downloaded with the repository.

## Controls

- Live confidence starts at 0.40.
- Live quality 256 is fastest, 320 is balanced, and 416 favors accuracy.
- Uploaded images use 640-pixel inference and start at 0.25 confidence.

The model is stored at `model/best.pt`. Retraining is not required to run the app.
