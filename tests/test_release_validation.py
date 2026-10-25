"""Dependency-light smoke tests for an offline research release."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]


def test_v5_protocol_hash_matches_embedded_digest():
    data = json.loads((root / "preregistration/PREREGISTRATION_V5.json").read_text())
    canonical = json.dumps(data["preregistration"], sort_keys=True, indent=2).encode()
    assert hashlib.sha256(canonical).hexdigest() == data["sha256"]


def test_v5_results_bind_protocol():
    protocol = json.loads((root / "preregistration/PREREGISTRATION_V5.json").read_text())
    record = json.loads((root / "results/V5_RESULTS.json").read_text())
    assert record["prereg_sha256"] == protocol["sha256"]


def test_stored_detector_corpus_coverage():
    record = json.loads((root / "results/V5_RESULTS.json").read_text())
    assert set(record["main"]) == {
        f"{det}|{corpus}" for det in ("sslaasist", "mamba", "aasist")
        for corpus in ("df21", "asv5")
    }


def test_external_artifacts_not_bundled():
    banned = {".pdf", ".tex", ".bib", ".pth", ".pt", ".ckpt", ".ipynb", ".wav", ".flac", ".zip"}
    assert not [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in banned]
