"""
Video capture manager with face detection.

Provides:
    - Context-managed webcam access (safe open/close)
    - Frame reading with error handling
    - Haar Cascade face detection (sorted by area, largest first)
    - Screenshot saving
"""

import os
import sys
from typing import List, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import AppConfig


class VideoCapture:
    """
    Managed webcam capture with built-in face detection.

    Usage
    -----
    >>> with VideoCapture(config) as cap:
    ...     frame = cap.read_frame()
    ...     faces = cap.detect_faces(frame)

    Parameters
    ----------
    config : AppConfig
        Application configuration.
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.cap: cv2.VideoCapture | None = None

        # Load Haar Cascade for face detection
        cascade_path = config.haar_cascade_path
        if not os.path.isfile(cascade_path):
            # Fall back to OpenCV's bundled cascade
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        if self.face_cascade.empty():
            raise RuntimeError(
                f"Failed to load Haar Cascade from: {cascade_path}\n"
                "Ensure OpenCV is installed correctly."
            )

    def __enter__(self) -> "VideoCapture":
        """Open the webcam."""
        self.cap = cv2.VideoCapture(self.config.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera at index {self.config.camera_index}.\n"
                "Check that your webcam is connected and not in use."
            )
        # Set resolution for performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print(f"[Camera] Opened camera {self.config.camera_index}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Release the webcam safely."""
        self.release()

    def release(self) -> None:
        """Release camera resources."""
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            print("[Camera] Released")
        self.cap = None

    def is_opened(self) -> bool:
        """Check if the camera is currently open."""
        return self.cap is not None and self.cap.isOpened()

    def read_frame(self) -> np.ndarray | None:
        """
        Read a single frame from the webcam.

        Returns
        -------
        np.ndarray or None
            BGR frame, or None if capture failed.
        """
        if not self.is_opened():
            return None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None

        return frame

    def detect_faces(
        self, frame: np.ndarray
    ) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in a frame using Haar Cascade.

        Parameters
        ----------
        frame : np.ndarray
            BGR input frame.

        Returns
        -------
        list of (x, y, w, h)
            Face bounding boxes sorted by area (largest first).
        """
        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Histogram equalization for better detection under varying lighting
        gray = cv2.equalizeHist(gray)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.config.face_scale_factor,
            minNeighbors=self.config.face_min_neighbors,
            minSize=(self.config.face_min_size, self.config.face_min_size),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        if len(faces) == 0:
            return []

        # Sort by area (largest first)
        face_list = [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]
        face_list.sort(key=lambda f: f[2] * f[3], reverse=True)

        return face_list

    def get_face_roi(
        self, frame: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """
        Extract a face region of interest from a frame.

        Parameters
        ----------
        frame : np.ndarray
            Full BGR frame.
        bbox : tuple
            (x, y, w, h) bounding box.

        Returns
        -------
        np.ndarray
            Cropped face region.
        """
        x, y, w, h = bbox
        # Clamp to frame boundaries
        x = max(0, x)
        y = max(0, y)
        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)
        return frame[y : y + h, x : x + w]

    def save_frame(self, frame: np.ndarray, path: str) -> bool:
        """
        Save a frame to disk as an image.

        Parameters
        ----------
        frame : np.ndarray
            Frame to save.
        path : str
            Output file path.

        Returns
        -------
        bool
            True if saved successfully.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        success = cv2.imwrite(path, frame)
        if success:
            print(f"[Camera] Screenshot saved: {path}")
        return success
