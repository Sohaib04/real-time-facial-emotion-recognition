"""
Main entry point for the Real-Time Emotion Detection application.

Orchestrates:
    - Config loading + CLI argument parsing
    - Model initialization (EmotionPredictor)
    - Webcam capture (VideoCapture)
    - Session logging (SessionLogger)
    - Display rendering (DisplayRenderer)
    - Prediction smoothing (EmotionSmoother)
    - Graceful shutdown with session summary + JSON report

Usage
-----
    python app/main.py
    python app/main.py --camera-index 1 --frame-skip 3 --use-gpu
    python app/main.py --confidence-threshold 0.6 --smoothing-window 7
"""

import os
import sys
import time
from datetime import datetime

import cv2

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import AppConfig, parse_args
from model.inference import EmotionPredictor
from app.video_capture import VideoCapture
from app.logger import SessionLogger
from app.display import DisplayRenderer
from app.emotion_smoother import EmotionSmoother


WINDOW_NAME = "Emotion Detection — Press Q to Quit"


def run(config: AppConfig) -> None:
    """
    Main application loop.

    Parameters
    ----------
    config : AppConfig
        Application configuration.
    """
    # ── Initialize components ──────────────────
    print("\n" + "=" * 55)
    print("  REAL-TIME EMOTION DETECTION")
    print("=" * 55)
    print(f"  Model       : {config.model_path}")
    print(f"  Camera      : index {config.camera_index}")
    print(f"  Frame skip  : {config.frame_skip}")
    print(f"  Threshold   : {config.confidence_threshold}")
    print(f"  GPU         : {config.use_gpu}")
    print(f"  Smoothing   : window={config.smoothing_window}")
    print("=" * 55 + "\n")

    predictor = EmotionPredictor(config)
    logger = SessionLogger(config)
    display = DisplayRenderer()
    smoother = EmotionSmoother(window_size=config.smoothing_window)

    # Track the last prediction for frame-skipping
    last_emotions = []  # list of (emotion, confidence) per face
    last_faces = []
    frame_count = 0

    try:
        with VideoCapture(config) as camera:
            print("[Main] Press 'Q' to quit, 'S' to save screenshot\n")

            while True:
                frame = camera.read_frame()
                if frame is None:
                    print("[WARN] Failed to read frame — retrying...")
                    time.sleep(0.1)
                    continue

                frame_count += 1

                # ── Face detection (every frame for smooth boxes) ─
                faces = camera.detect_faces(frame)

                # ── Prediction (every N frames for performance) ───
                if frame_count % config.frame_skip == 0 or not last_emotions:
                    emotions = []
                    for bbox in faces:
                        face_roi = camera.get_face_roi(frame, bbox)
                        if face_roi.size == 0:
                            continue

                        emotion, confidence = predictor.predict(face_roi)

                        # Apply smoothing
                        smoothed_emotion, smoothed_conf = smoother.update(
                            emotion, confidence
                        )

                        # Apply confidence threshold
                        if smoothed_conf >= config.confidence_threshold:
                            emotions.append((smoothed_emotion, smoothed_conf))
                            logger.log_prediction(smoothed_emotion, smoothed_conf)
                        else:
                            emotions.append(("Uncertain", smoothed_conf))

                    last_emotions = emotions
                    last_faces = faces
                else:
                    # Reuse last predictions for skipped frames
                    # Still use current face positions for box rendering
                    faces_to_show = faces[:len(last_emotions)] if faces else last_faces
                    emotions = last_emotions[:len(faces_to_show)]
                    faces = faces_to_show

                # ── Render overlays ────────────────────────────
                frame = display.render(frame, faces, last_emotions[:len(faces)])

                # ── Display ────────────────────────────────────
                cv2.imshow(WINDOW_NAME, frame)

                # ── Key handling ───────────────────────────────
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q") or key == ord("Q"):
                    print("\n[Main] Quit requested")
                    break

                elif key == ord("s") or key == ord("S"):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    screenshot_path = os.path.join(
                        config.screenshot_dir,
                        f"screenshot_{timestamp}.png",
                    )
                    camera.save_frame(frame, screenshot_path)

    except KeyboardInterrupt:
        print("\n[Main] Interrupted by user")
    except RuntimeError as e:
        print(f"\n[ERROR] {e}")
    finally:
        # ── Cleanup ────────────────────────────────
        cv2.destroyAllWindows()

        # Session summary
        summary = logger.end_session()

        # Save JSON report
        report_path = logger.save_report()

        logger.close()

        print(f"\n[Main] Session ended. Report: {report_path}")
        print("[Main] Goodbye!\n")


def main() -> None:
    """Parse CLI args and launch the application."""
    config = parse_args()
    run(config)


if __name__ == "__main__":
    main()
