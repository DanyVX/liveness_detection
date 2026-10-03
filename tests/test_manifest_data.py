import csv
from pathlib import Path

import pytest

from liveness.data import Label, ManifestError, Split, load_manifest_dataset


def _write_manifest(root: Path, *, bad_hash: bool = False) -> Path:
    rows = []
    for split_index, split in enumerate(("train", "val", "test")):
        for label, attack in (("live", "none"), ("attack", "print")):
            subject = f"s{split_index}{label[0]}"
            path = root / f"{subject}.jpg"
            path.write_bytes(f"{subject}-{label}".encode())
            rows.append(
                {
                    "relative_path": path.name,
                    "subject_id": subject,
                    "label": label,
                    "attack_type": attack,
                    "split": split,
                    "sha256": "0" * 64 if bad_hash else "",
                }
            )
    manifest = root / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def test_manifest_loads_and_builds_protocol(tmp_path: Path) -> None:
    _write_manifest(tmp_path)
    dataset = load_manifest_dataset("licensed", tmp_path)
    assert len(dataset.samples) == 6
    assert {sample.label for sample in dataset.samples} == {Label.ATTACK, Label.BONA_FIDE}
    protocol = dataset.protocol("official")
    assert protocol.name == "official"
    assert all(len(protocol.splits[split]) == 2 for split in Split)


@pytest.mark.parametrize("relative", ["../outside.jpg", "C:/outside.jpg"])
def test_manifest_rejects_unsafe_paths(tmp_path: Path, relative: str) -> None:
    outside = tmp_path.parent / "outside.jpg"
    outside.write_bytes(b"x")
    (tmp_path / "manifest.csv").write_text(
        f"relative_path,subject_id,label,attack_type,split\n{relative},s,live,none,train\n"
    )
    with pytest.raises(ManifestError, match="dataset root"):
        load_manifest_dataset("d", tmp_path)


def test_manifest_rejects_bad_checksum(tmp_path: Path) -> None:
    _write_manifest(tmp_path, bad_hash=True)
    with pytest.raises(ManifestError, match="SHA-256 mismatch"):
        load_manifest_dataset("d", tmp_path)


def test_manifest_requires_all_columns(tmp_path: Path) -> None:
    (tmp_path / "manifest.csv").write_text("relative_path,subject_id\nfoo,s\n")
    with pytest.raises(ManifestError, match="missing columns"):
        load_manifest_dataset("d", tmp_path)
