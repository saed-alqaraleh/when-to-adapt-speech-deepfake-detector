"""Pilot runner + gate verdict on synthetic inputs (CPU). Run: python -m pytest -q tests/test_gates.py"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pga_g import pilot as P  # noqa: E402
from pga_g import gates  # noqa: E402
from pga_g.runner import pilot_stage  # noqa: E402


def test_pilot_runner_and_verdict(tmp_path):
    ws = tmp_path / "PGA_G"
    pin = ws / "results" / "pilot_inputs"
    pin.mkdir(parents=True)
    df = P.synthetic_scores(np.random.default_rng(3), n_bona=3000, n_spoof=3000, hard_mean=0.0, hard_frac=0.5)
    df.to_csv(pin / "itw_pilot.csv", index=False)
    np.save(pin / "itw_emb160.npy", P.synthetic_embeddings(df, np.random.default_rng(4)).astype(np.float32))
    (ws / "results" / "G0_reproduction.json").write_text(json.dumps({"G0": {"pass": True}}))

    pilot_stage("stageA", ws, ROOT, ["--scores", str(pin / "itw_pilot.csv"), "--source-threshold", "4.97",
                                     "--reps", "2", "--jobs", "2"], [pin / "itw_pilot.csv"])
    pilot_stage("stageB_emb160", ws, ROOT, ["--scores", str(pin / "itw_pilot.csv"), "--embeddings",
                                            str(pin / "itw_emb160.npy"), "--stage-b", "--reps-b", "2",
                                            "--jobs", "2"], [pin / "itw_pilot.csv", pin / "itw_emb160.npy"])
    m1 = (ws / "results" / "stageA" / "_STAGE_SUCCESS.json").stat().st_mtime
    pilot_stage("stageA", ws, ROOT, ["--scores", str(pin / "itw_pilot.csv"), "--source-threshold", "4.97",
                                     "--reps", "2", "--jobs", "2"], [pin / "itw_pilot.csv"])
    assert (ws / "results" / "stageA" / "_STAGE_SUCCESS.json").stat().st_mtime == m1   # skipped

    assert gates.main(["verdict", "--workspace", str(ws)]) == 0
    out = json.loads((ws / "results" / "GATES.json").read_text())
    assert out["decision"].split()[0] in {"GO", "PIVOT", "INVESTIGATE", "REVIEW"}
    assert {"G2", "G3", "G4"} <= set(out["stageB_emb160"])
