<div align="center">

# 🎮 RPS Vision Arena

### Play with your hands. Let AI call the move.

**✊ Rock · ✋ Paper · ✌️ Scissors**

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Live_App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-AI_Vision-00FFFF?style=for-the-badge)
![OpenCV](https://img.shields.io/badge/OpenCV-Camera-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)

Turn your webcam into a colorful Rock–Paper–Scissors arena. Challenge a
friend, strike a pose, and let computer vision decide who wins! 🏆

</div>

Use the live webcam, snap a photo, or upload an image. In **Battle Mode**, the
app recognizes Player 1 on the left and Player 2 on the right, reads both hand
gestures, and announces the winner directly on the video.

## ✨ What makes it fun?

- 🪞 **Mirrored live camera** — move naturally, just like looking in a mirror
- ⏱️ **3–2–1 countdown** — get ready, reveal your move, and hold it!
- 🧠 **Five-frame AI voting** — helps avoid calling a move while you are moving
- ⚔️ **Two-player Battle Mode** — automatic P1/P2 labels and winner calculation
- 📸 **Three ways to play** — live camera, camera snapshot, or uploaded image
- 🎛️ **Adjustable controls** — tune confidence and quality for your computer
- 🫧 **Glassmorphism design** — colorful cards, glowing borders, and fullscreen play
- ☁️ **Run anywhere** — macOS, Windows, or Streamlit Community Cloud

## 🥊 The rules of the arena

| Move | Beats | Why? |
|---|---|---|
| ✊ **Rock** | ✌️ Scissors | Rock crushes scissors |
| ✋ **Paper** | ✊ Rock | Paper covers rock |
| ✌️ **Scissors** | ✋ Paper | Scissors cut paper |

## 🚀 Ready… set… SHOW!

1. 📹 Start the camera and allow browser camera access.
2. ⚔️ Enable **Two-player Battle Mode** when challenging a friend.
3. 🩷 Player 1 takes the left side; 🩵 Player 2 takes the right.
4. ▶️ Press **START ROUND**.
5. ⏳ Wait for `3… 2… 1…`
6. 🙌 Reveal and hold your gesture when the screen says **SHOW!**
7. 🏆 Let the AI read the moves and crown the winner.

If a gesture is not detected consistently, the app asks the players to try the
round again instead of intentionally forcing a low-confidence result.

## 🗂️ Behind the arena

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

## 💻 Launch the game locally

### 🍎 macOS

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

### 🪟 Windows / VS Code

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

## ☁️ Put the arena online

1. Push all project files, including `model/best.pt`, to GitHub.
2. Visit [Streamlit Community Cloud](https://share.streamlit.io/).
3. Select this repository and the `main` branch.
4. Set the application entry point to `app.py`.
5. Deploy the application and allow camera access when prompted.

`packages.txt` provides the Linux libraries required by OpenCV. The WebRTC
configuration includes a public STUN server for remote camera connections.

## 🎛️ Arena controls

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

## 🧠 Meet the AI referee

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

## 🛠️ Built with

- Python
- Streamlit
- streamlit-webrtc
- Ultralytics YOLOv8
- OpenCV
- PyTorch

## 💡 Tips for a championship-level round

- Keep both hands fully visible and away from the camera edges.
- Face gestures toward the camera and hold them still after **SHOW!**
- Use bright, even lighting and leave space between both players.
- Local performance is normally faster than free cloud hosting.
- A CUDA-capable computer requires CUDA-enabled PyTorch to use an NVIDIA GPU.
  Apple Silicon uses PyTorch MPS when available.

---

<div align="center">

### Ready to enter the arena?

## ✊ ✋ ✌️

**Start the camera. Challenge a friend. May the best hand win!**

</div>
