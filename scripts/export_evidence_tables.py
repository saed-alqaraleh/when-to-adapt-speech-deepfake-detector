"""Export reproducible study-level numeric evidence datasets, not publication tables.

Each CSV is a deterministic grouped mean of the released per-repetition records.
Historical study names remain distinct. No rounding or inferred confidence intervals.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GROUP_FIELDS = ("ref_mode", "mode", "pi", "n_ref", "n_traffic", "detector", "corpus",
                "design", "method", "variant", "attack", "poison", "fraction",
                "profile", "batch_size", "window")
MEASURES = ("eer", "hter", "frr", "far", "shortcut_auc", "h_eer", "h_hter",
            "h_frr", "h_far", "gate_open", "adapted", "pi_hat_frozen")
STUDIES = {
    "anchor": ("stageA_df21/results.csv",),
    "first_gate": ("pga_df21/results.csv",),
    "guard": ("robust_df21/results.csv", "robust_itw/results.csv"),
    "asv5_confirmation": ("asv5_main/results.csv",),
    "detector_generalisation": tuple(f"v3/main_{c}_{d}/results.csv" for d in
                                  ("sslaasist", "mamba", "aasist") for c in ("df21", "asv5")),
    "deployment_robustness": tuple(f"v4/{kind}_{c}_{d}/results.csv" for kind in
                                   ("agnostic", "edge", "poison") for d in
                                   ("sslaasist", "mamba", "aasist") for c in ("df21", "asv5")),
    "source_group_audit": tuple(f"v5/main_{c}_{d}/results.csv" for d in
                                ("sslaasist", "mamba", "aasist") for c in ("df21", "asv5")),
}


def aggregate(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    keys = [x for x in GROUP_FIELDS if x in df.columns]
    values = [x for x in MEASURES if x in df.columns]
    if not keys or not values or "rep" not in df.columns:
        raise ValueError(f"Missing group/measurement/repetition columns in {path}")
    if df[keys + values].isna().any().any():
        raise ValueError(f"Null group or outcome in {path}")
    frame = df.groupby(keys, dropna=False, sort=True)[values].agg(["mean", "count"])
    frame.columns = ["_".join(col) for col in frame.columns]
    frame = frame.reset_index()
    frame.insert(0, "source_file", path.relative_to(ROOT).as_posix())
    return frame


def export(destination: Path) -> dict[str, int]:
    destination.mkdir(parents=True, exist_ok=True)
    counts = {}
    for study, members in STUDIES.items():
        tables = []
        for member in members:
            file = ROOT / "results" / member
            if not file.exists():
                raise FileNotFoundError(file)
            tables.append(aggregate(file))
        table = pd.concat(tables, ignore_index=True, sort=False)
        table.to_csv(destination / f"{study}.csv", index=False, float_format="%.17g")
        counts[study] = len(table)
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="Destination outside the tracked research evidence")
    args = parser.parse_args()
    for name, count in export(args.output).items():
        print(f"{name}: {count} condition-level rows")
