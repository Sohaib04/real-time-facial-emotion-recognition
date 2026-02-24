"""
OpenCV overlay renderer for the real-time display.

Draws:
    - Colored bounding boxes per emotion
    - Emotion label + confidence percentage
    - FPS counter (rolling average)
    - Session timer
    - Status messages and key hints
"""

import os
import sys
import time
from collections import deque
from typing import List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import EMOTION_COLORS


class DisplayRenderer:
    """
    Renders detection overlays on video frames.

    Parameters
    ----------
    fps_window : int
        Number of frames for rolling FPS average.
    """

    # Font settings
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX
    FONT_SCALE_LABEL = 0.7
    FONT_SCALE_SMALL = 0.5
    FONT_THICKNESS = 2
    BOX_THICKNESS = 2

    def __init__(self, fps_window: int = 30) -> None:
        self.frame_times: deque = deque(maxlen=fps_window)
        self.session_start = time.time()

    def tick(self) -> None:
        """Record a frame timestamp for FPS calculation."""
        self.frame_times.append(time.time())

    @property
    def fps(self) -> float:
        """Calculate the current rolling-average FPS."""
        if len(self.frame_times) < 2:
            return 0.0
        elapsed = self.frame_times[-1] - self.frame_times[0]
        if elapsed <= 0:
            return 0.0
        return (len(self.frame_times) - 1) / elapsed

    @property
    def session_elapsed(self) -> float:
        """Seconds since session start."""
        return time.time() - self.session_start

    def draw_face_box(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        emotion: str,
        confidence: float,
    ) -> np.ndarray:
        """
        Draw a colored bounding box with emotion label and confidence.

        Parameters
        ----------
        frame : np.ndarray
            BGR frame to draw on (modified in-place).
        bbox : tuple
            (x, y, w, h) face bounding box.
        emotion : str
            Predicted emotion label.
        confidence : float
            Confidence score (0-1).

        Returns
        -------
        np.ndarray
            Frame with overlay drawn.
        """
        x, y, w, h = bbox
        color = EMOTION_COLORS.get(emotion, (255, 255, 255))

        # Bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, self.BOX_THICKNESS)

        # Label background
        label = f"{emotion} {confidence:.0%}"
        (text_w, text_h), baseline = cv2.getTextSize(
            label, self.FONT, self.FONT_SCALE_LABEL, self.FONT_THICKNESS
        )
        label_y = max(y - 10, text_h + 10)
        cv2.rectangle(
            frame,
            (x, label_y - text_h - 6),
            (x + text_w + 8, label_y + 4),
            color,
            cv2.FILLED,
        )

        # Label text (black on colored background for readability)
        cv2.putText(
            frame,
            label,
            (x + 4, label_y - 2),
            self.FONT,
            self.FONT_SCALE_LABEL,
            (0, 0, 0),
            self.FONT_THICKNESS,
            cv2.LINE_AA,
        )

        return frame

    def draw_hud(
        self,
        frame: np.ndarray,
        status_message: Optional[str] = None,
    ) -> np.ndarray:
        """
        Draw the heads-up display: FPS, session timer, key hints.

        Parameters
        ----------
        frame : np.ndarray
            BGR frame to draw on (modified in-place).
        status_message : str or None
            Optional status message (e.g., "No face detected").

        Returns
        -------
        np.ndarray
            Frame with HUD drawn.
        """
        h, w = frame.shape[:2]

        # ── Top-left: FPS counter ──────────────────
        fps_text = f"FPS: {self.fps:.1f}"
        cv2.putText(
            frame, fps_text, (10, 28),
            self.FONT, self.FONT_SCALE_SMALL, (0, 255, 0), 1, cv2.LINE_AA,
        )

        # ── Top-left: Session timer ───────────────
        elapsed = self.session_elapsed
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        timer_text = f"Session: {minutes:02d}:{seconds:02d}"
        cv2.putText(
            frame, timer_text, (10, 52),
            self.FONT, self.FONT_SCALE_SMALL, (0, 255, 0), 1, cv2.LINE_AA,
        )

        # ── Bottom-left: Key hints ────────────────
        hints = "Q: Quit  |  S: Screenshot"
        cv2.putText(
            frame, hints, (10, h - 15),
            self.FONT, self.FONT_SCALE_SMALL, (180, 180, 180), 1, cv2.LINE_AA,
        )

        # ── Center: Status message ────────────────
        if status_message:
            (msg_w, msg_h), _ = cv2.getTextSize(
                status_message, self.FONT, self.FONT_SCALE_LABEL, self.FONT_THICKNESS
            )
            msg_x = (w - msg_w) // 2
            msg_y = (h + msg_h) // 2

            # Semi-transparent background
            overlay = frame.copy()
            cv2.rectangle(
                overlay,
                (msg_x - 12, msg_y - msg_h - 12),
                (msg_x + msg_w + 12, msg_y + 12),
                (0, 0, 0),
                cv2.FILLED,
            )
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            cv2.putText(
                frame, status_message, (msg_x, msg_y),
                self.FONT, self.FONT_SCALE_LABEL, (0, 200, 255), self.FONT_THICKNESS, cv2.LINE_AA,
            )

        return frame

    def draw_no_face(self, frame: np.ndarray) -> np.ndarray:
        """Draw a 'No face detected' indicator."""
        return self.draw_hud(frame, status_message="No face detected")

    def render(
        self,
        frame: np.ndarray,
        faces: List[Tuple[int, int, int, int]],
        emotions: List[Tuple[str, float]],
    ) -> np.ndarray:
        """
        Full render pass: draw all face boxes and HUD.

        Parameters
        ----------
        frame : np.ndarray
            BGR frame.
        faces : list of (x, y, w, h)
            Detected face bounding boxes.
        emotions : list of (emotion_label, confidence)
            Corresponding predictions for each face.

        Returns
        -------
        np.ndarray
            Rendered frame.
        """
        self.tick()

        if not faces:
            return self.draw_no_face(frame)

        for bbox, (emotion, conf) in zip(faces, emotions):
            self.draw_face_box(frame, bbox, emotion, conf)

        return self.draw_hud(frame)
