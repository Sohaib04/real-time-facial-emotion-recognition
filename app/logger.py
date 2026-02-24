"""
Session logger backed by SQLite.

Provides:
    - Automatic session creation with UUID
    - Per-prediction logging (timestamp, emotion, confidence)
    - Session summary computation (dominant emotion, frequencies, avg confidence)
    - JSON session report export
"""

import json
import os
import sqlite3
import sys
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import EMOTION_LABELS, AppConfig


class SessionLogger:
    """
    SQLite-backed emotion prediction logger.

    Creates a database with two tables:
        - sessions: session metadata (id, start/end time)
        - predictions: per-frame predictions (timestamp, emotion, confidence)

    Parameters
    ----------
    config : AppConfig
        Application configuration.
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.session_id = str(uuid.uuid4())[:8]
        self.start_time = datetime.now()

        db_path = os.path.join(config.log_dir, config.db_name)
        os.makedirs(config.log_dir, exist_ok=True)

        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_tables()
        self._insert_session()

        self.prediction_count = 0
        print(f"[Logger] Session {self.session_id} started — DB: {db_path}")

    def _create_tables(self) -> None:
        """Create the sessions and predictions tables if they don't exist."""
        cursor = self.conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                start_time TEXT NOT NULL,
                end_time TEXT,
                dominant_emotion TEXT,
                total_predictions INTEGER DEFAULT 0,
                avg_confidence REAL DEFAULT 0.0
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                emotion TEXT NOT NULL,
                confidence REAL NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions (session_id)
            )
        """)

        # Index for fast session queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_predictions_session
            ON predictions (session_id)
        """)

        self.conn.commit()

    def _insert_session(self) -> None:
        """Insert the session record."""
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO sessions (session_id, start_time) VALUES (?, ?)",
            (self.session_id, self.start_time.isoformat()),
        )
        self.conn.commit()

    def log_prediction(self, emotion: str, confidence: float) -> None:
        """
        Log a single prediction.

        Parameters
        ----------
        emotion : str
            Predicted emotion label.
        confidence : float
            Prediction confidence (0-1).
        """
        timestamp = datetime.now().isoformat()
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO predictions (session_id, timestamp, emotion, confidence) "
            "VALUES (?, ?, ?, ?)",
            (self.session_id, timestamp, emotion, confidence),
        )
        self.conn.commit()
        self.prediction_count += 1

    def get_session_summary(self) -> Dict[str, Any]:
        """
        Compute summary statistics for the current session.

        Returns
        -------
        dict
            Session summary containing:
                - session_id
                - start_time, end_time, duration_seconds
                - total_predictions
                - dominant_emotion
                - emotion_frequencies (dict)
                - avg_confidence
        """
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()

        cursor = self.conn.cursor()

        # Emotion frequency counts
        cursor.execute(
            "SELECT emotion, COUNT(*) FROM predictions "
            "WHERE session_id = ? GROUP BY emotion ORDER BY COUNT(*) DESC",
            (self.session_id,),
        )
        rows = cursor.fetchall()
        emotion_frequencies = {row[0]: row[1] for row in rows}

        # Average confidence
        cursor.execute(
            "SELECT AVG(confidence) FROM predictions WHERE session_id = ?",
            (self.session_id,),
        )
        avg_confidence = cursor.fetchone()[0] or 0.0

        # Dominant emotion
        dominant_emotion = rows[0][0] if rows else "None"

        return {
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": round(duration, 2),
            "total_predictions": self.prediction_count,
            "dominant_emotion": dominant_emotion,
            "emotion_frequencies": emotion_frequencies,
            "avg_confidence": round(avg_confidence, 4),
        }

    def end_session(self) -> Dict[str, Any]:
        """
        Finalize the session, update the database, and print a summary.

        Returns
        -------
        dict
            Session summary.
        """
        summary = self.get_session_summary()

        # Update session record
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE sessions SET end_time = ?, dominant_emotion = ?, "
            "total_predictions = ?, avg_confidence = ? WHERE session_id = ?",
            (
                summary["end_time"],
                summary["dominant_emotion"],
                summary["total_predictions"],
                summary["avg_confidence"],
                self.session_id,
            ),
        )
        self.conn.commit()

        # Print summary
        self._print_summary(summary)

        return summary

    def save_report(self, path: Optional[str] = None) -> str:
        """
        Save the session summary as a JSON report.

        Parameters
        ----------
        path : str or None
            Output path. If None, auto-generates in log_dir.

        Returns
        -------
        str
            Path to the saved report.
        """
        summary = self.get_session_summary()

        if path is None:
            filename = f"session_{self.session_id}_{self.start_time.strftime('%Y%m%d_%H%M%S')}.json"
            path = os.path.join(self.config.log_dir, filename)

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"[Logger] Session report saved: {path}")
        return path

    def close(self) -> None:
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def _print_summary(self, summary: Dict[str, Any]) -> None:
        """Pretty-print session summary to terminal."""
        print("\n" + "=" * 55)
        print("  SESSION SUMMARY")
        print("=" * 55)
        print(f"  Session ID       : {summary['session_id']}")
        print(f"  Duration         : {summary['duration_seconds']:.1f}s")
        print(f"  Total Predictions: {summary['total_predictions']}")
        print(f"  Dominant Emotion : {summary['dominant_emotion']}")
        print(f"  Avg Confidence   : {summary['avg_confidence']:.2%}")
        print("-" * 55)
        print("  Emotion Breakdown:")
        for emotion, count in summary["emotion_frequencies"].items():
            bar_len = int(count / max(summary["total_predictions"], 1) * 30)
            bar = "█" * bar_len
            print(f"    {emotion:10s} {count:4d}  {bar}")
        print("=" * 55 + "\n")
