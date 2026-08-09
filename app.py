from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
from ultralytics import YOLO


MODEL_PATH = Path(__file__).parent / "model" / "best.pt"
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = YOLO(str(MODEL_PATH))
DEVICE = 0 if torch.cuda.is_available() else "cpu"

# Warm up the model once so the first webcam frame does not pay startup cost.
model.predict(
    source=np.zeros((320, 320, 3), dtype=np.uint8),
    imgsz=320,
    device=DEVICE,
    verbose=False,
)


def run_detection(image, confidence, image_size):
    """Run detection on an RGB image and return an annotated RGB image."""
    if image is None:
        return None

    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    result = model.predict(
        source=image_bgr,
        conf=float(confidence),
        imgsz=image_size,
        device=DEVICE,
        max_det=10,
        verbose=False,
    )[0]
    annotated_bgr = result.plot()
    return cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)


def detect_webcam(image, confidence):
    """Use a smaller input size to keep live detection responsive."""
    return run_detection(image, confidence, image_size=320)


def detect_uploaded_image(image, confidence):
    """Use a larger input size for better uploaded-image accuracy."""
    return run_detection(image, confidence, image_size=640)


with gr.Blocks(title="Rock Paper Scissors Detection") as demo:
    gr.Markdown(
        """
        # Rock-Paper-Scissors Detection
        Detects **paper**, **rock**, and **scissors** hand gestures.
        """
    )

    with gr.Tab("Live camera"):
        live_confidence = gr.Slider(
            minimum=0.10,
            maximum=0.90,
            value=0.40,
            step=0.05,
            label="Live confidence threshold",
        )
        with gr.Row():
            webcam = gr.Image(
                sources=["webcam"],
                type="numpy",
                streaming=True,
                label="Camera",
            )
            live_output = gr.Image(type="numpy", label="Live detections")

        webcam.stream(
            fn=detect_webcam,
            inputs=[webcam, live_confidence],
            outputs=live_output,
            # Run directly instead of allowing webcam frames to build up in a queue.
            queue=False,
            stream_every=0.50,
            trigger_mode="always_last",
        )

    with gr.Tab("Upload image"):
        upload_confidence = gr.Slider(
            minimum=0.10,
            maximum=0.90,
            value=0.25,
            step=0.05,
            label="Upload confidence threshold",
        )
        with gr.Row():
            uploaded_image = gr.Image(type="numpy", label="Input image")
            image_output = gr.Image(type="numpy", label="Detections")
        detect_button = gr.Button("Detect", variant="primary")
        detect_button.click(
            fn=detect_uploaded_image,
            inputs=[uploaded_image, upload_confidence],
            outputs=image_output,
        )


if __name__ == "__main__":
    print(f"Inference device: {DEVICE}")
    print(f"Model classes: {model.names}")
    demo.launch(inbrowser=True, share=False)
