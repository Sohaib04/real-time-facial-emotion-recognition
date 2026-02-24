"""
Emotion prediction smoother.

Applies a moving-average over the last N predictions to reduce frame-to-frame
jitter and provide a more stable emotion readout.

Also tracks the rolling dominant emotion across the full session.
"""

import os
import sys
from collections import Counter, deque
from typing import Dict, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import EMOTION_LABELS


class EmotionSmoother:
    """
    Smooths emotion predictions using a sliding window of recent results.

    Parameters
    ----------
    window_size : int
        Number of recent predictions to average over.
    """

    def __init__(self, window_size: int = 5) -> None:
        self.window_size = window_size

        # Sliding window of (emotion_label, confidence) tuples
        self.history: deque = deque(maxlen=window_size)

        # Full-session emotion counter for rolling dominant tracking
        self.session_counter: Counter = Counter()

    def update(self, emotion: str, confidence: float) -> Tuple[str, float]:
        """
        Add a new prediction and return the smoothed result.

        The smoothed emotion is determined by majority vote over the window.
        The smoothed confidence is the average confidence of the majority emotion
        within the window.

        Parameters
        ----------
        emotion : str
            Raw predicted emotion.
        confidence : float
            Raw prediction confidence (0-1).

        Returns
        -------
        tuple[str, float]
            (smoothed_emotion, smoothed_confidence)
        """
        self.history.append((emotion, confidence))
        self.session_counter[emotion] += 1

        # Count emotions in the window
        window_emotions = [e for e, _ in self.history]
        window_counts = Counter(window_emotions)

        # Majority emotion
        smoothed_emotion = window_counts.most_common(1)[0][0]

        # Average confidence for the majority emotion within the window
        majority_confidences = [
            c for e, c in self.history if e == smoothed_emotion
        ]
        smoothed_confidence = sum(majority_confidences) / len(majority_confidences)

        return smoothed_emotion, smoothed_confidence

    @property
    def dominant_emotion(self) -> Optional[str]:
        """
        The most frequently predicted emotion across the entire session.

        Returns
        -------
        str or None
            The dominant emotion label, or None if no predictions yet.
        """
        if not self.session_counter:
            return None
        return self.session_counter.most_common(1)[0][0]

    @property
    def emotion_distribution(self) -> Dict[str, int]:
        """
        Full emotion frequency distribution for the session.

        Returns
        -------
        dict[str, int]
            Emotion → count mapping.
        """
        return dict(self.session_counter)

    def reset(self) -> None:
        """Clear the sliding window and session counter."""
        self.history.clear()
        self.session_counter.clear()
