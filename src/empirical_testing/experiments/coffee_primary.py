"""Raw Coffee snapshot diagnostics used by the maintained experiment suite."""

from __future__ import annotations

import hashlib
from pathlib import Path


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def describe_data_hash_mismatch(data_path: Path, expected_sha256: str) -> str:
    """Explain a raw-data hash mismatch, distinguishing converted line endings."""

    raw = Path(data_path).read_bytes()
    crlf = raw.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    if _sha256_bytes(crlf) == expected_sha256:
        return (
            f"{data_path} has the frozen content but its line endings were converted (the frozen "
            "snapshot uses CRLF). The repository marks this file '-text' in .gitattributes; restore "
            "the original bytes with `git checkout HEAD -- data/coffee/raw/coffee_survey.csv` or re-clone."
        )
    return f"{data_path} does not match the frozen Coffee survey snapshot (sha256 {expected_sha256})"
