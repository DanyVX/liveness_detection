"""Validated manifest adapter for licensed PAD datasets.

Raw layouts differ by dataset release and may change.  This adapter makes the conversion
explicit: an owner creates ``manifest.csv`` beside the licensed files, then every downstream
component consumes the same audited columns without guessing identities or protocols.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

from liveness.data.base import AttackType, Label, Sample
from liveness.data.protocols import Protocol, Split

REQUIRED_COLUMNS = {
    "relative_path",
    "subject_id",
    "label",
    "attack_type",
    "split",
}


class ManifestError(ValueError):
    """A dataset manifest is unsafe, incomplete, or internally inconsistent."""


@dataclass(frozen=True, slots=True)
class ManifestDataset:
    name: str
    samples: tuple[Sample, ...]
    split_by_path: dict[Path, Split]

    def protocol(self, name: str | None = None) -> Protocol:
        return Protocol(
            name=name or f"{self.name}-manifest",
            splits={
                split: tuple(s for s in self.samples if self.split_by_path[s.path] is split)
                for split in Split
            },
        )


def _safe_path(root: Path, relative: str, row: int) -> Path:
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise ManifestError(f"row {row}: relative_path must stay under the dataset root")
    root_resolved = root.resolve()
    path = (root / rel).resolve()
    if path != root_resolved and root_resolved not in path.parents:
        raise ManifestError(f"row {row}: relative_path escapes the dataset root")
    if not path.is_file():
        raise ManifestError(f"row {row}: file does not exist: {relative}")
    return path


def _label(value: str, row: int) -> Label:
    normalized = value.strip().lower()
    if normalized in {"1", "bona_fide", "bonafide", "live", "real"}:
        return Label.BONA_FIDE
    if normalized in {"0", "attack", "spoof", "fake"}:
        return Label.ATTACK
    raise ManifestError(f"row {row}: unknown label {value!r}")


def _attack_type(value: str, label: Label, row: int) -> AttackType:
    normalized = value.strip().lower()
    aliases = {"live": "none", "real": "none", "photo": "print", "video": "replay"}
    normalized = aliases.get(normalized, normalized)
    try:
        attack = AttackType(normalized)
    except ValueError as exc:
        raise ManifestError(f"row {row}: unknown attack_type {value!r}") from exc
    if (label is Label.BONA_FIDE) != (attack is AttackType.NONE):
        raise ManifestError(f"row {row}: label and attack_type disagree")
    return attack


def _split(value: str, row: int) -> Split:
    normalized = {"dev": "val", "devel": "val", "validation": "val"}.get(
        value.strip().lower(), value.strip().lower()
    )
    try:
        return Split(normalized)
    except ValueError as exc:
        raise ManifestError(f"row {row}: unknown split {value!r}") from exc


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest_dataset(
    name: str,
    root: Path,
    manifest_path: Path | None = None,
    *,
    verify_checksums: bool = True,
) -> ManifestDataset:
    """Load a normalized manifest and enforce files, labels, checksums and split disjointness."""
    manifest = manifest_path or root / "manifest.csv"
    if not manifest.is_file():
        raise ManifestError(f"manifest not found: {manifest}; see DATA.md")
    samples: list[Sample] = []
    split_by_path: dict[Path, Split] = {}
    seen: set[Path] = set()
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ManifestError(f"manifest is missing columns: {sorted(missing)}")
        for row_number, raw in enumerate(reader, start=2):
            path = _safe_path(root, raw["relative_path"].strip(), row_number)
            if path in seen:
                raise ManifestError(f"row {row_number}: duplicate file {raw['relative_path']!r}")
            seen.add(path)
            label = _label(raw["label"], row_number)
            sample = Sample(
                dataset=name,
                subject_id=raw["subject_id"].strip(),
                path=path,
                label=label,
                attack_type=_attack_type(raw["attack_type"], label, row_number),
            )
            expected = (raw.get("sha256") or "").strip().lower()
            if verify_checksums and expected:
                actual = sha256_file(path)
                if actual != expected:
                    raise ManifestError(
                        f"row {row_number}: SHA-256 mismatch for {raw['relative_path']!r}"
                    )
            samples.append(sample)
            split_by_path[path] = _split(raw["split"], row_number)
    if not samples:
        raise ManifestError("manifest contains no samples")
    dataset = ManifestDataset(name, tuple(samples), split_by_path)
    dataset.protocol()  # Validate subject/file disjointness and both classes per split now.
    return dataset


def write_checksum_manifest(root: Path, paths: list[Path], output: Path) -> None:
    """Write deterministic ``SHA256  relative/path`` lines for locally licensed files."""
    root = root.resolve()
    rows: list[str] = []
    for path in sorted((p.resolve() for p in paths), key=lambda p: p.as_posix()):
        if root not in path.parents:
            raise ManifestError(f"checksum target is outside root: {path}")
        rows.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(rows) + "\n", encoding="utf-8")
