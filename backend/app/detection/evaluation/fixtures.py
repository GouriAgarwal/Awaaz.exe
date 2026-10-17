"""Unit test fixtures and mock dataset generator.

IMPORTANT:
These fixtures are designed solely for unit test verification of software pipelines,
file parsing, and metrics calculations.
They must NEVER be presented or evaluated as real synthetic speech benchmarks.
"""

import csv
import io
from pathlib import Path
from typing import Tuple
import numpy as np
import soundfile as sf

from app.detection.evaluation.dataset import DatasetManifest, DatasetManifestParser


def create_mock_test_wav(
    filepath: Path,
    duration_sec: float = 1.0,
    freq: float = 440.0,
    sample_rate: int = 16000,
) -> None:
    """Generate in-memory sinusoidal test audio and write to disk for software unit tests."""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    samples = (0.4 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(filepath), samples, sample_rate, format="WAV")


def generate_mock_test_dataset(
    base_dir: Path,
    num_real: int = 3,
    num_synthetic: int = 3,
) -> Tuple[Path, DatasetManifest]:
    """Create a temporary mock CSV dataset manifest and audio files for unit testing.
    
    Returns:
        Tuple of (manifest_csv_path, DatasetManifest)
    """
    audio_dir = base_dir / "audio_samples"
    audio_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    # Create REAL mock audio files
    for i in range(num_real):
        filename = f"real_sample_{i+1:03d}.wav"
        audio_path = audio_dir / filename
        create_mock_test_wav(audio_path, duration_sec=1.2, freq=200.0 + i * 50)
        rows.append({
            "file": str(audio_path.resolve()),
            "label": "REAL",
            "split": "test",
            "speaker_id": f"spk_real_{i+1}",
            "generator": "human_vocal_tract",
            "language": "en",
            "accent": "general",
            "codec": "wav",
            "noise_condition": "clean",
        })

    # Create SYNTHETIC mock audio files
    for i in range(num_synthetic):
        filename = f"synthetic_sample_{i+1:03d}.wav"
        audio_path = audio_dir / filename
        create_mock_test_wav(audio_path, duration_sec=1.5, freq=600.0 + i * 100)
        rows.append({
            "file": str(audio_path.resolve()),
            "label": "SYNTHETIC",
            "split": "test",
            "speaker_id": f"spk_synth_{i+1}",
            "generator": "mock_vocoder_v1",
            "language": "en",
            "accent": "general",
            "codec": "wav",
            "noise_condition": "clean",
        })

    manifest_csv = base_dir / "test_manifest.csv"
    with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    manifest = DatasetManifestParser.parse_manifest_file(manifest_csv)
    return manifest_csv, manifest
