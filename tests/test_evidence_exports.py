"""Smoke test for the per-repetition evidence exporter."""
from pathlib import Path

import pandas as pd

from scripts.export_evidence_tables import export


def test_condition_exports_are_recomputable(tmp_path):
    counts = export(tmp_path)
    assert set(counts) == {
        "anchor", "first_gate", "guard", "asv5_confirmation",
        "detector_generalisation", "deployment_robustness", "source_group_audit",
    }
    for name, count in counts.items():
        p = tmp_path / (name + ".csv")
        table = pd.read_csv(p)
        assert len(table) == count > 0
        assert "source_file" in table
        assert not table.filter(regex="_mean$").isna().all().all()
    baseline = (tmp_path / "source_group_audit.csv").read_bytes()
    assert export(tmp_path) == counts
    assert (tmp_path / "source_group_audit.csv").read_bytes() == baseline
