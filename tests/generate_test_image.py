"""
Generate a synthetic test face image for unit testing.

Creates a simple 48×48 grayscale image with basic facial features.
"""

import os

import cv2
import numpy as np


def generate_test_face(output_path: str) -> None:
    """Create a synthetic face-like image for testing."""
    img = np.full((48, 48), 180, dtype=np.uint8)  # Light gray background

    # Simple "face" using circles and ellipses
    # Face oval
    cv2.ellipse(img, (24, 26), (18, 22), 0, 0, 360, 140, -1)

    # Eyes
    cv2.circle(img, (16, 20), 3, 60, -1)  # Left eye
    cv2.circle(img, (32, 20), 3, 60, -1)  # Right eye

    # Nose
    cv2.line(img, (24, 24), (22, 30), 100, 1)

    # Mouth (smile)
    cv2.ellipse(img, (24, 34), (8, 4), 0, 0, 180, 80, 1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img)
    print(f"Test face image saved: {output_path}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output = os.path.join(script_dir, "assets", "test_face.png")
    generate_test_face(output)
