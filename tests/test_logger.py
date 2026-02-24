"""
Unit tests for the SessionLogger.

Tests:
    - Session creation with UUID
    - Prediction insertion
    - Summary statistics computation
    - JSON report generation
    - Database table structure

Uses temporary databases for isolated testing.
"""

import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import AppConfig
from app.logger import SessionLogger


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────
@pytest.fixture
def temp_config(tmp_path):
    """Create an AppConfig with a temporary log directory."""
    config = AppConfig()
    config.log_dir = str(tmp_path)
    config.db_name = "test_emotion_log.db"
    return config


@pytest.fixture
def logger(temp_config):
    """Create a SessionLogger backed by a temporary database."""
    lg = SessionLogger(temp_config)
    yield lg
    lg.close()


# ──────────────────────────────────────────────
# Session Creation Tests
# ──────────────────────────────────────────────
class TestSessionCreation:
    """Test session initialization."""

    def test_session_id_generated(self, logger):
        """Session should have a non-empty UUID-based ID."""
        assert logger.session_id
        assert len(logger.session_id) == 8

    def test_session_id_is_string(self, logger):
        """Session ID should be a string."""
        assert isinstance(logger.session_id, str)

    def test_start_time_set(self, logger):
        """Start time should be set on creation."""
        assert logger.start_time is not None

    def test_initial_prediction_count(self, logger):
        """Prediction count should start at 0."""
        assert logger.prediction_count == 0

    def test_database_file_created(self, temp_config, logger):
        """Database file should exist after logger creation."""
        db_path = os.path.join(temp_config.log_dir, temp_config.db_name)
        assert os.path.isfile(db_path)


# ──────────────────────────────────────────────
# Prediction Logging Tests
# ──────────────────────────────────────────────
class TestPredictionLogging:
    """Test prediction insertion."""

    def test_log_single_prediction(self, logger):
        """Should log a single prediction and increment count."""
        logger.log_prediction("Happy", 0.95)
        assert logger.prediction_count == 1

    def test_log_multiple_predictions(self, logger):
        """Should track count across multiple predictions."""
        for i in range(10):
            logger.log_prediction("Neutral", 0.80)
        assert logger.prediction_count == 10

    def test_log_different_emotions(self, logger):
        """Should handle different emotion types."""
        logger.log_prediction("Angry", 0.70)
        logger.log_prediction("Happy", 0.90)
        logger.log_prediction("Sad", 0.65)
        assert logger.prediction_count == 3

    def test_log_low_confidence(self, logger):
        """Should accept low-confidence predictions."""
        logger.log_prediction("Fear", 0.01)
        assert logger.prediction_count == 1

    def test_log_max_confidence(self, logger):
        """Should accept maximum-confidence predictions."""
        logger.log_prediction("Surprise", 1.0)
        assert logger.prediction_count == 1


# ──────────────────────────────────────────────
# Summary Statistics Tests
# ──────────────────────────────────────────────
class TestSessionSummary:
    """Test session summary computation."""

    def test_summary_structure(self, logger):
        """Summary should contain all expected keys."""
        logger.log_prediction("Happy", 0.9)
        summary = logger.get_session_summary()

        expected_keys = {
            "session_id", "start_time", "end_time",
            "duration_seconds", "total_predictions",
            "dominant_emotion", "emotion_frequencies", "avg_confidence",
        }
        assert expected_keys == set(summary.keys())

    def test_dominant_emotion(self, logger):
        """Dominant emotion should be the most frequent."""
        for _ in range(5):
            logger.log_prediction("Happy", 0.9)
        for _ in range(2):
            logger.log_prediction("Sad", 0.6)

        summary = logger.get_session_summary()
        assert summary["dominant_emotion"] == "Happy"

    def test_emotion_frequencies(self, logger):
        """Frequencies should match logged predictions."""
        logger.log_prediction("Happy", 0.8)
        logger.log_prediction("Happy", 0.9)
        logger.log_prediction("Angry", 0.7)

        summary = logger.get_session_summary()
        freqs = summary["emotion_frequencies"]
        assert freqs["Happy"] == 2
        assert freqs["Angry"] == 1

    def test_average_confidence(self, logger):
        """Average confidence should be correctly computed."""
        logger.log_prediction("Happy", 0.8)
        logger.log_prediction("Happy", 0.6)
        logger.log_prediction("Happy", 1.0)

        summary = logger.get_session_summary()
        assert abs(summary["avg_confidence"] - 0.8) < 0.01

    def test_empty_session_summary(self, logger):
        """Empty session should have 0 predictions and None-like dominant."""
        summary = logger.get_session_summary()
        assert summary["total_predictions"] == 0
        assert summary["dominant_emotion"] == "None"

    def test_total_predictions_matches(self, logger):
        """Total predictions in summary should match count."""
        for _ in range(15):
            logger.log_prediction("Neutral", 0.75)
        summary = logger.get_session_summary()
        assert summary["total_predictions"] == 15


# ──────────────────────────────────────────────
# End Session Tests
# ──────────────────────────────────────────────
class TestEndSession:
    """Test session finalization."""

    def test_end_session_returns_summary(self, logger):
        """end_session should return a valid summary dict."""
        logger.log_prediction("Fear", 0.5)
        summary = logger.end_session()
        assert isinstance(summary, dict)
        assert "session_id" in summary

    def test_end_session_updates_db(self, logger, temp_config):
        """Database should have end_time after end_session."""
        import sqlite3
        logger.log_prediction("Happy", 0.9)
        logger.end_session()

        db_path = os.path.join(temp_config.log_dir, temp_config.db_name)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT end_time FROM sessions WHERE session_id = ?",
            (logger.session_id,),
        )
        end_time = cursor.fetchone()[0]
        conn.close()
        assert end_time is not None


# ──────────────────────────────────────────────
# JSON Report Tests
# ──────────────────────────────────────────────
class TestJsonReport:
    """Test JSON report generation."""

    def test_report_file_created(self, logger, tmp_path):
        """JSON report file should be created."""
        logger.log_prediction("Happy", 0.85)
        path = logger.save_report(str(tmp_path / "test_report.json"))
        assert os.path.isfile(path)

    def test_report_is_valid_json(self, logger, tmp_path):
        """Report file should contain valid JSON."""
        logger.log_prediction("Sad", 0.7)
        path = logger.save_report(str(tmp_path / "test_report.json"))
        with open(path) as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_report_contains_summary(self, logger, tmp_path):
        """Report should contain session summary data."""
        logger.log_prediction("Happy", 0.9)
        logger.log_prediction("Neutral", 0.8)
        path = logger.save_report(str(tmp_path / "test_report.json"))
        with open(path) as f:
            data = json.load(f)
        assert data["total_predictions"] == 2
        assert "emotion_frequencies" in data

    def test_auto_generated_path(self, logger):
        """Report should auto-generate path in log_dir."""
        logger.log_prediction("Happy", 0.9)
        path = logger.save_report()
        assert os.path.isfile(path)
        assert logger.config.log_dir in path
