"""Dataset manifest parsing, validation, and speaker leakage analysis for voice clone evaluations."""

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from app.audio.decoder import AudioDecoder
from app.detection.evaluation.schemas import DatasetManifest, DatasetSample


class DatasetValidationError(Exception):
    """Exception raised when dataset manifest parsing or sample validation fails."""
    pass


class DatasetManifestParser:
    """Parser and validator for evaluation dataset manifests (CSV or JSON)."""

    VALID_LABELS = {"REAL", "SYNTHETIC"}

    @staticmethod
    def _compute_file_hash(filepath: Path) -> str:
        """Compute SHA256 hash of file content to detect duplicate audio payloads."""
        hasher = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    @classmethod
    def parse_manifest_file(
        cls,
        manifest_path: str | Path,
        base_dir: Optional[str | Path] = None,
        verify_audio_decoding: bool = True,
    ) -> DatasetManifest:
        """Parse, validate, and load an evaluation dataset manifest.
        
        Args:
            manifest_path: Path to CSV or JSON manifest file.
            base_dir: Optional base directory to resolve relative audio filepaths.
            verify_audio_decoding: If True, tests in-memory audio decoding for each sample.
            
        Returns:
            Validated DatasetManifest instance.
            
        Raises:
            DatasetValidationError if manifest cannot be parsed or contains critical errors.
        """
        path = Path(manifest_path)
        if not path.exists():
            raise DatasetValidationError(f"Manifest file not found: {path}")

        resolved_base = Path(base_dir) if base_dir else path.parent
        dataset_name = path.stem

        if path.suffix.lower() == ".csv":
            raw_entries = cls._parse_csv(path)
        elif path.suffix.lower() in [".json", ".jsonl"]:
            raw_entries = cls._parse_json(path)
        else:
            raise DatasetValidationError(f"Unsupported manifest format '{path.suffix}'. Expected .csv or .json.")

        if not raw_entries:
            raise DatasetValidationError(f"Manifest '{path.name}' is empty (0 records found).")

        validated_samples: List[DatasetSample] = []
        seen_file_paths: Set[str] = set()
        seen_content_hashes: Dict[str, str] = {}
        real_count = 0
        synthetic_count = 0
        split_counts: Dict[str, int] = {}
        generator_counts: Dict[str, int] = {}
        speaker_ids: Set[str] = set()

        for idx, entry in enumerate(raw_entries, start=1):
            raw_file = entry.get("file") or entry.get("audio_path") or entry.get("path")
            if not raw_file:
                raise DatasetValidationError(f"Row {idx}: Missing 'file' audio path.")

            # Resolve absolute filepath
            audio_path = Path(raw_file)
            if not audio_path.is_absolute():
                audio_path = resolved_base / audio_path

            if not audio_path.exists():
                raise DatasetValidationError(f"Row {idx}: Audio file does not exist: {audio_path}")

            # Check duplicate file paths
            canonical_path = str(audio_path.resolve())
            if canonical_path in seen_file_paths:
                raise DatasetValidationError(f"Row {idx}: Duplicate audio file path: {canonical_path}")
            seen_file_paths.add(canonical_path)

            # Check duplicate content hashes
            content_hash = cls._compute_file_hash(audio_path)
            if content_hash in seen_content_hashes:
                prev_file = seen_content_hashes[content_hash]
                raise DatasetValidationError(
                    f"Row {idx}: Duplicate audio content detected. '{audio_path.name}' is identical to '{prev_file}'"
                )
            seen_content_hashes[content_hash] = audio_path.name

            # Validate ground truth label
            raw_label = str(entry.get("label", "")).strip().upper()
            if raw_label not in cls.VALID_LABELS:
                raise DatasetValidationError(
                    f"Row {idx}: Invalid label '{raw_label}'. Must be exactly 'REAL' or 'SYNTHETIC'."
                )

            # Optional audio decoding verification
            if verify_audio_decoding:
                try:
                    with open(audio_path, "rb") as f:
                        file_bytes = f.read()
                    if len(file_bytes) == 0:
                        raise DatasetValidationError(f"Row {idx}: Audio file is empty (0 bytes): {audio_path.name}")
                    samples, _, _, _ = AudioDecoder.decode(file_bytes)
                    if len(samples) == 0 or not np.all(np.isfinite(samples)):
                        raise DatasetValidationError(
                            f"Row {idx}: Decoded audio has zero samples or non-finite numbers: {audio_path.name}"
                        )
                except Exception as exc:
                    if isinstance(exc, DatasetValidationError):
                        raise
                    raise DatasetValidationError(f"Row {idx}: Audio decoding failed for '{audio_path.name}': {exc}")

            # Collect metadata
            split = str(entry.get("split", "test")).strip().lower()
            speaker_id = str(entry.get("speaker_id")).strip() if entry.get("speaker_id") else None
            generator = str(entry.get("generator")).strip() if entry.get("generator") else None
            language = str(entry.get("language")).strip() if entry.get("language") else None
            accent = str(entry.get("accent")).strip() if entry.get("accent") else None
            gender = str(entry.get("gender")).strip() if entry.get("gender") else None
            codec = str(entry.get("codec")).strip() if entry.get("codec") else audio_path.suffix.lstrip(".").lower()
            noise = str(entry.get("noise_condition") or entry.get("noise")).strip() if (entry.get("noise_condition") or entry.get("noise")) else None

            if raw_label == "REAL":
                real_count += 1
            else:
                synthetic_count += 1

            split_counts[split] = split_counts.get(split, 0) + 1
            if generator:
                generator_counts[generator] = generator_counts.get(generator, 0) + 1
            if speaker_id:
                speaker_ids.add(speaker_id)

            sample = DatasetSample(
                file=str(audio_path.resolve()),
                label=raw_label,
                split=split,
                speaker_id=speaker_id,
                generator=generator,
                language=language,
                accent=accent,
                gender=gender,
                codec=codec,
                noise_condition=noise,
            )
            validated_samples.append(sample)

        return DatasetManifest(
            dataset_name=dataset_name,
            samples=validated_samples,
            total_samples=len(validated_samples),
            real_count=real_count,
            synthetic_count=synthetic_count,
            split_counts=split_counts,
            speaker_count=len(speaker_ids),
            generator_counts=generator_counts,
        )

    @classmethod
    def check_speaker_leakage(cls, manifest: DatasetManifest) -> Tuple[bool, List[str]]:
        """Analyze dataset for potential speaker identity leakage across splits or classes.
        
        Returns:
            Tuple of (has_leakage: bool, leakage_descriptions: List[str])
        """
        leakage_notes: List[str] = []
        has_leakage = False

        # Group speakers by split
        speakers_by_split: Dict[str, Set[str]] = {}
        for s in manifest.samples:
            if s.speaker_id and s.split:
                speakers_by_split.setdefault(s.split, set()).add(s.speaker_id)

        splits = list(speakers_by_split.keys())
        for i in range(len(splits)):
            for j in range(i + 1, len(splits)):
                s1, s2 = splits[i], splits[j]
                overlap = speakers_by_split[s1].intersection(speakers_by_split[s2])
                if overlap:
                    has_leakage = True
                    leakage_notes.append(
                        f"Speaker leakage across splits: {len(overlap)} speakers appear in both '{s1}' and '{s2}'."
                    )

        return has_leakage, leakage_notes

    @classmethod
    def _parse_csv(cls, path: Path) -> List[Dict[str, str]]:
        """Parse CSV manifest into dictionary rows."""
        entries: List[Dict[str, str]] = []
        with open(path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise DatasetValidationError(f"CSV file '{path.name}' is empty or missing headers.")
            for row in reader:
                cleaned_row = {k.strip(): (v.strip() if v else "") for k, v in row.items() if k}
                entries.append(cleaned_row)
        return entries

    @classmethod
    def _parse_json(cls, path: Path) -> List[Dict[str, str]]:
        """Parse JSON or JSONL manifest."""
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                raise DatasetValidationError(f"JSON manifest '{path.name}' is empty.")
            if content.startswith("["):
                data = json.loads(content)
                if not isinstance(data, list):
                    raise DatasetValidationError(f"JSON manifest must contain an array of sample records.")
                return data
            else:
                # JSON Lines format
                entries = []
                for line in content.splitlines():
                    if line.strip():
                        entries.append(json.loads(line))
                return entries
