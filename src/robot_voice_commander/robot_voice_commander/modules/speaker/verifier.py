# Speaker verifier module for the Robot Voice Commander system.

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

class SpeakerVerifier:
    """
    Verifies if an audio sample belongs to an authorized speaker.

    The system stores one voice embedding per authorized speaker.
    When a new command is recorded, a new embedding is created and
    compared against the stored ones.
    """

    def __init__(self, config: dict) -> None:
        self._enabled: bool = config.get("enabled", True)
        self._speakers_dir = Path(config.get("speakers_dir", "authorized_speakers"))
        self._threshold: float = float(config.get("threshold", 0.75))
        self._max_speakers: int = int(config.get("max_speakers", 2))
        self._sample_rate: int = int(config.get("sample_rate", 16000))
        self._num_bands: int = int(config.get("num_bands", 32))

        self._speakers_dir.mkdir(parents=True, exist_ok=True)
        self._metadata_path = self._speakers_dir / "speakers.json"

        logger.info(
            "SpeakerVerifier initialized (enabled=%s, threshold=%.2f)",
            self._enabled,
            self._threshold,
        )

    def is_enabled(self) -> bool:
        """Returns True if speaker verification is enabled."""
        return self._enabled

    def list_speakers(self) -> list[str]:
        """Returns the names of the enrolled speakers."""
        metadata = self._load_metadata()
        return list(metadata.keys())

    def enroll_speaker(self, name: str, audio: np.ndarray) -> bool:
        """
        Registers a new authorized speaker.

        Args:
            name: Speaker name.
            audio: Recorded audio as a numpy array.

        Returns:
            True if the speaker was enrolled correctly.
            False if the speaker could not be enrolled.
        """
        if not name:
            logger.error("Speaker name cannot be empty")
            return False

        metadata = self._load_metadata()

        if name not in metadata and len(metadata) >= self._max_speakers:
            logger.error(
                "Maximum number of speakers reached (%d)",
                self._max_speakers,
            )
            return False

        embedding = self._create_embedding(audio)

        if embedding is None:
            logger.error("Could not create speaker embedding")
            return False

        file_name = f"{name}.npy"
        file_path = self._speakers_dir / file_name

        np.save(file_path, embedding)

        metadata[name] = {
            "file": file_name,
            "sample_rate": self._sample_rate,
        }

        self._save_metadata(metadata)

        logger.info("Speaker '%s' enrolled successfully", name)
        return True

    def verify(self, audio: np.ndarray) -> tuple[bool, Optional[str], float]:
        """
        Checks if the given audio matches any authorized speaker.

        Args:
            audio: Recorded audio as a numpy array.

        Returns:
            authorized: True if a speaker matched.
            speaker_name: Name of the matched speaker, or None.
            score: Best similarity score.
        """
        if not self._enabled:
            return True, "verification_disabled", 1.0

        metadata = self._load_metadata()

        if not metadata:
            logger.warning("No authorized speakers enrolled")
            return False, None, 0.0

        current_embedding = self._create_embedding(audio)

        if current_embedding is None:
            logger.warning("Could not create embedding from current audio")
            return False, None, 0.0

        best_name = None
        best_score = -1.0

        for speaker_name, info in metadata.items():
            embedding_path = self._speakers_dir / info["file"]

            if not embedding_path.exists():
                logger.warning("Missing embedding file: %s", embedding_path)
                continue

            stored_embedding = np.load(embedding_path)
            score = self._cosine_similarity(current_embedding, stored_embedding)

            logger.info(
                "Speaker comparison with '%s': score=%.3f",
                speaker_name,
                score,
            )

            if score > best_score:
                best_score = score
                best_name = speaker_name

        authorized = best_score >= self._threshold

        if authorized:
            logger.info(
                "Authorized speaker detected: %s (score=%.3f)",
                best_name,
                best_score,
            )
        else:
            logger.warning(
                "Unauthorized speaker (best=%s, score=%.3f)",
                best_name,
                best_score,
            )

        return authorized, best_name, float(best_score)

    def _create_embedding(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """
        Creates a simple voice fingerprint from audio.

        This method:
        1. Removes very low amplitude sections.
        2. Splits audio into frames.
        3. Computes frequency energy.
        4. Groups the spectrum into frequency bands.
        5. Averages all frames into one vector.
        """
        if audio is None or len(audio) == 0:
            return None

        audio = np.asarray(audio, dtype=np.float32)

        # Normalize audio amplitude
        max_value = np.max(np.abs(audio))
        if max_value < 1e-6:
            return None

        audio = audio / max_value

        # Remove very quiet samples
        audio = audio[np.abs(audio) > 0.01]

        if len(audio) < self._sample_rate * 0.5:
            logger.warning("Audio too short for speaker verification")
            return None

        frame_size = 1024
        hop_size = 512

        features = []

        for start in range(0, len(audio) - frame_size, hop_size):
            frame = audio[start:start + frame_size]

            # Window to reduce spectral artifacts
            window = np.hanning(frame_size)
            frame = frame * window

            spectrum = np.abs(np.fft.rfft(frame))

            # Avoid DC component
            spectrum = spectrum[1:]

            band_features = self._group_frequency_bands(spectrum)
            features.append(band_features)

        if not features:
            return None

        embedding = np.mean(np.array(features), axis=0)

        # Normalize embedding
        norm = np.linalg.norm(embedding)
        if norm < 1e-6:
            return None

        return embedding / norm

    def _group_frequency_bands(self, spectrum: np.ndarray) -> np.ndarray:
        """
        Groups the frequency spectrum into a fixed number of bands.
        """
        bands = np.array_split(spectrum, self._num_bands)
        band_energy = np.array([np.mean(band) for band in bands], dtype=np.float32)

        # Log scale makes the feature less sensitive to amplitude changes
        return np.log1p(band_energy)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Computes cosine similarity between two embeddings.
        """
        denominator = np.linalg.norm(a) * np.linalg.norm(b)

        if denominator < 1e-6:
            return 0.0

        return float(np.dot(a, b) / denominator)

    def _load_metadata(self) -> dict:
        """
        Loads speaker metadata from speakers.json.
        """
        if not self._metadata_path.exists():
            return {}

        with open(self._metadata_path, "r", encoding="utf-8") as file:
            return json.load(file)

    def _save_metadata(self, metadata: dict) -> None:
        """
        Saves speaker metadata to speakers.json.
        """
        with open(self._metadata_path, "w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)