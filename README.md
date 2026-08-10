# Rock Paper Scissors — RPS Vision Arena

An interactive Rock–Paper–Scissors computer-vision game built with
**Streamlit**, **WebRTC**, and a custom **YOLOv8** object-detection model.

Players can use a live webcam, take a photo, or upload an image. In two-player
battle mode, the app identifies Player 1 on the left and Player 2 on the right,
then announces the winner on the video.

## Features

- Live, mirrored webcam experience
- Three-second `3–2–1` game countdown
- Five-frame voting after the countdown to reduce predictions during movement
- Two-player battle mode with automatic winner calculation
- Separate camera-photo and image-upload modes
- Adjustable detection confidence and inference quality
- Custom glassmorphism interface with responsive fullscreen video
- Local macOS, Windows, and Streamlit Community Cloud support

## How a live round works

1. Start the camera and allow browser camera access.
2. Enable **Two-player battle mode** if two people are playing.
3. Player 1 stands on the left and Player 2 stands on the right.
4. Press **START ROUND**.
5. Reveal and hold each gesture when the countdown reaches **SHOW!**
6. The app checks five quick frames and uses repeated predictions to determine
   the result.

If a gesture is not detected consistently, the app asks the players to try the
round again instead of intentionally forcing a low-confidence result.

## Project structure

```text
Rock-Paper-Scissors-cv/
├── app.py                         # Streamlit interface and detection logic
├── model/
│   └── best.pt                    # Trained YOLO model weights
├── notebooks/
│   └── training_notebook.ipynb    # Google Colab training workflow
├── dataset.yaml                   # Dataset class and path configuration
├── requirements.txt               # Python dependencies
├── packages.txt                   # Linux packages for Streamlit Cloud
├── run_mac.command                # macOS setup and launcher
├── start_class.command            # Fast macOS launcher after setup
├── .gitignore
└── README.md
```

## Run locally

### macOS

Open Terminal inside the project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

For later launches:

```bash
source .venv/bin/activate
python -m streamlit run app.py
```

You can also double-click `run_mac.command`. After the first successful setup,
use `start_class.command` for a faster classroom launch.

### Windows / VS Code

Open the repository folder in VS Code, select **Terminal → New Terminal**, and
run:

```powershell
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open `http://localhost:8501` if the browser does not open automatically.

## Deploy on Streamlit Community Cloud

1. Push all project files, including `model/best.pt`, to GitHub.
2. Visit [Streamlit Community Cloud](https://share.streamlit.io/).
3. Select this repository and the `main` branch.
4. Set the application entry point to `app.py`.
5. Deploy the application and allow camera access when prompted.

`packages.txt` provides the Linux libraries required by OpenCV. The WebRTC
configuration includes a public STUN server for remote camera connections.

## Controls and performance

- **Confidence:** Raising it rejects more uncertain detections; lowering it
  detects more hands but may produce more incorrect predictions.
- **Live quality:** `256` is fastest, `320` is balanced, and `416` provides more
  detail at the cost of speed.
- **Mirror camera:** Makes movement feel natural and keeps player positions
  intuitive.
- **Battle mode:** Treats the left gesture as Player 1 and the right gesture as
  Player 2.

For the clearest results, use even lighting, keep both hands fully inside the
frame, leave space between players, and hold gestures facing the camera.

## Model

The YOLO model recognizes these classes:

```text
0: paper
1: rock
2: scissors
3: unknown
```

The included weights are located at `model/best.pt`. No training is required
to launch the application.

Model accuracy depends on how closely webcam conditions resemble the training
dataset. The countdown and voting system can stabilize predictions, but they
cannot correct a class the model consistently misunderstands. For better
real-world accuracy, retrain with images captured from the intended camera and
environment, including different people, lighting, distances, hand angles,
backgrounds, and movement/unknown examples.

After retraining, replace `model/best.pt` with the new best weights and restart
the application.

## Technology

- Python
- Streamlit
- streamlit-webrtc
- Ultralytics YOLOv8
- OpenCV
- PyTorch

## Notes

- Camera processing occurs through the browser and Streamlit WebRTC session.
- Local performance is normally faster than free cloud hosting.
- A CUDA-capable computer requires a CUDA-enabled PyTorch installation to use
  an NVIDIA GPU. Apple Silicon uses PyTorch MPS when available.
