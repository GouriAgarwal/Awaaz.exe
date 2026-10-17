"""Dataset Discovery, Integrity Audit, and Leakage Analysis Subsystem (Phase 11A)."""

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from app.audio.decoder import AudioDecoder
from app.core.logging import get_logger
from app.evaluation.schemas import DatasetAuditReport, DatasetAvailabilityStatus

logger = get_logger("evaluation.dataset_audit")

# Known benchmark dataset directories/manifest signatures to search for
KNOWN_DATASET_PATTERNS = [
    "asvspoof2019_la",
    "asvspoof2021_la",
    "asvspoof2021_df",
    "wavefake",
    "fake_or_real",
    "for_dataset",
    "voxceleb1",
    "in_the_wild",
]


class DatasetAuditError(Exception):
    """Exception raised during dataset manifest audit."""
    pass


class DatasetAuditor:
    """Rigorous dataset discovery, integrity auditing, and leakage analyzer."""

    VALID_LABELS = {"REAL", "SYNTHETIC", "BONAFIDE", "SPOOF", "TARGET", "NON_TARGET", "GENUINE", "IMPOSTOR"}
    REAL_SYNONYMS = {"REAL", "BONAFIDE", "GENUINE", "TARGET"}
    SYNTHETIC_SYNONYMS = {"SYNTHETIC", "SPOOF", "IMPOSTOR", "NON_TARGET"}

    @staticmethod
    def compute_sha256(filepath: Path) -> str:
        """Compute SHA-256 hash of a file."""
        hasher = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    @classmethod
    def discover_local_datasets(cls, search_root: Optional[Path] = None) -> List[Path]:
        """Search workspace directories for recognized benchmark manifests or datasets."""
        root = search_root or Path(__file__).resolve().parents[3]
        discovered: List[Path] = []

        for ext in ["*.csv", "*.json", "*.jsonl"]:
            for manifest in root.glob(f"**/{ext}"):
                # Exclude node_modules, cache, git
                path_str = str(manifest).lower()
                if any(x in path_str for x in ["node_modules", ".venv", "__pycache__", ".git", "package"]):
                    continue
                # Check if name or parent indicates a known dataset
                if any(p in path_str for p in KNOWN_DATASET_PATTERNS) or "manifest" in path_str or "dataset" in path_str:
                    discovered.append(manifest)

        return discovered

    @classmethod
    def audit_manifest(
        cls,
        manifest_path: str | Path,
        base_dir: Optional[str | Path] = None,
        verify_audio_decoding: bool = True,
        max_samples_to_decode: int = 500,
    ) -> DatasetAuditReport:
        """Perform exhaustive integrity audit of a dataset manifest.
        
        Audits:
        1. File existence & non-empty size
        2. SHA-256 duplicate content detection
        3. Valid label detection
        4. Split presence & distribution
        5. Cross-split speaker identity leakage
        6. Generator leakage across splits/classes
        7. Optional audio decoding verification
        """
        path = Path(manifest_path)
        if not path.exists():
            return DatasetAuditReport(
                dataset_name=path.stem,
                status=DatasetAvailabilityStatus.NOT_AVAILABLE_LOCALLY,
                manifest_path=str(path),
                integrity_notes=[f"Manifest file not found at '{path}'."],
            )

        resolved_base = Path(base_dir) if base_dir else path.parent
        dataset_name = path.stem

        # Parse manifest records
        try:
            raw_entries = cls._parse_entries(path)
        except Exception as e:
            return DatasetAuditReport(
                dataset_name=dataset_name,
                status=DatasetAvailabilityStatus.CORRUPTED_OR_INVALID,
                manifest_path=str(path),
                integrity_notes=[f"Failed to parse manifest syntax: {e}"],
            )

        if not raw_entries:
            return DatasetAuditReport(
                dataset_name=dataset_name,
                status=DatasetAvailabilityStatus.CORRUPTED_OR_INVALID,
                manifest_path=str(path),
                integrity_notes=["Manifest contains 0 records."],
            )

        seen_file_paths: Set[str] = set()
        seen_content_hashes: Dict[str, str] = {}
        missing_files: List[str] = []
        duplicate_files: List[str] = []
        integrity_notes: List[str] = []
        
        real_count = 0
        synthetic_count = 0
        split_counts: Dict[str, int] = {}
        generator_counts: Dict[str, int] = {}
        speakers_by_split: Dict[str, Set[str]] = {}
        generators_by_class: Dict[str, Set[str]] = {}
        all_speakers: Set[str] = set()

        decoded_count = 0

        for idx, entry in enumerate(raw_entries, start=1):
            raw_file = entry.get("file") or entry.get("audio_path") or entry.get("path")
            if not raw_file:
                integrity_notes.append(f"Row {idx}: Missing audio path column.")
                continue

            audio_path = Path(raw_file)
            if not audio_path.is_absolute():
                audio_path = resolved_base / audio_path

            if not audio_path.exists():
                missing_files.append(str(audio_path))
                continue

            # Check duplicate file path
            canonical = str(audio_path.resolve())
            if canonical in seen_file_paths:
                duplicate_files.append(canonical)
                continue
            seen_file_paths.add(canonical)

            # Check duplicate content hash
            try:
                content_hash = cls.compute_sha256(audio_path)
                if content_hash in seen_content_hashes:
                    duplicate_files.append(f"{audio_path.name} (identical to {seen_content_hashes[content_hash]})")
                else:
                    seen_content_hashes[content_hash] = audio_path.name
            except Exception as e:
                integrity_notes.append(f"Row {idx}: Failed to compute hash for '{audio_path.name}': {e}")

            # Validate label
            raw_label = str(entry.get("label", "")).strip().upper()
            if raw_label not in cls.VALID_LABELS:
                integrity_notes.append(f"Row {idx}: Invalid label '{raw_label}'.")
                canonical_label = "UNKNOWN"
            elif raw_label in cls.REAL_SYNONYMS:
                canonical_label = "REAL"
                real_count += 1
            else:
                canonical_label = "SYNTHETIC"
                synthetic_count += 1

            # Metadata fields
            split = str(entry.get("split", "test")).strip().lower()
            split_counts[split] = split_counts.get(split, 0) + 1

            speaker_id = str(entry.get("speaker_id")).strip() if entry.get("speaker_id") else None
            if speaker_id:
                all_speakers.add(speaker_id)
                speakers_by_split.setdefault(split, set()).add(speaker_id)

            generator = str(entry.get("generator")).strip() if entry.get("generator") else None
            if generator:
                generator_counts[generator] = generator_counts.get(generator, 0) + 1
                generators_by_class.setdefault(canonical_label, set()).add(generator)

            # Optional decode test
            if verify_audio_decoding and decoded_count < max_samples_to_decode:
                try:
                    with open(audio_path, "rb") as f:
                        file_bytes = f.read()
                    if len(file_bytes) == 0:
                        integrity_notes.append(f"Row {idx}: 0-byte audio file '{audio_path.name}'.")
                    else:
                        samples, _, _, _ = AudioDecoder.decode(file_bytes)
                        if len(samples) == 0 or not np.all(np.isfinite(samples)):
                            integrity_notes.append(f"Row {idx}: Decoded audio contains NaNs/Infs: '{audio_path.name}'.")
                    decoded_count += 1
                except Exception as e:
                    integrity_notes.append(f"Row {idx}: Audio decode failed for '{audio_path.name}': {e}")

        # Speaker leakage analysis across splits
        has_speaker_leakage = False
        speaker_leakage_notes: List[str] = []
        splits = list(speakers_by_split.keys())
        for i in range(len(splits)):
            for j in range(i + 1, len(splits)):
                s1, s2 = splits[i], splits[j]
                overlap = speakers_by_split[s1].intersection(speakers_by_split[s2])
                if overlap:
                    has_speaker_leakage = True
                    speaker_leakage_notes.append(
                        f"Speaker leakage across splits: {len(overlap)} speakers shared between '{s1}' and '{s2}'."
                    )

        # Generator leakage / bias reporting
        has_generator_leakage = False
        generator_leakage_notes: List[str] = []
        if "REAL" in generators_by_class and generators_by_class["REAL"]:
            has_generator_leakage = True
            generator_leakage_notes.append(
                f"Generator tags assigned to REAL speech ({len(generators_by_class['REAL'])} items). REAL speech should be natural."
            )

        # Determine overall dataset status
        if missing_files and len(missing_files) == len(raw_entries):
            status = DatasetAvailabilityStatus.NOT_AVAILABLE_LOCALLY
            integrity_notes.append("All audio files in manifest are missing from local filesystem.")
        elif missing_files or integrity_notes or has_speaker_leakage:
            status = DatasetAvailabilityStatus.AVAILABLE_WITH_WARNINGS
        else:
            status = DatasetAvailabilityStatus.AVAILABLE_VALIDATED

        return DatasetAuditReport(
            dataset_name=dataset_name,
            status=status,
            manifest_path=str(path),
            total_samples=len(raw_entries) - len(missing_files),
            real_count=real_count,
            synthetic_count=synthetic_count,
            split_counts=split_counts,
            speaker_count=len(all_speakers),
            generator_counts=generator_counts,
            has_speaker_leakage=has_speaker_leakage,
            speaker_leakage_notes=speaker_leakage_notes,
            has_generator_leakage=has_generator_leakage,
            generator_leakage_notes=generator_leakage_notes,
            duplicate_count=len(duplicate_files),
            duplicate_files=duplicate_files,
            missing_files=missing_files,
            integrity_notes=integrity_notes,
        )

    @classmethod
    def _parse_entries(cls, path: Path) -> List[Dict[str, str]]:
        if path.suffix.lower() == ".csv":
            with open(path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames:
                    return []
                return [{k.strip(): (v.strip() if v else "") for k, v in row.items() if k} for row in reader]
        elif path.suffix.lower() in [".json", ".jsonl"]:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    return []
                if content.startswith("["):
                    return json.loads(content)
                else:
                    return [json.loads(line) for line in content.splitlines() if line.strip()]
        return []
