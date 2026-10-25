# PGA-G: Selective Adaptation for Speech Deepfake Detection

**Original PGA-G methodology, research implementation, and reproducible experimental evidence**

PGA-G is a selective adaptation method developed for this study. Its implementation decides whether to retain or adapt a frozen speech-deepfake detector using an unlabelled deployment batch and a genuine-speech reference. The study-specific pipeline combines a prevalence gate, a consistency guard that screens for reference-dependent shortcuts, and an independently calibrated conformal operating point under the corresponding calibration design. It is evaluated using SSL-AASIST, XLSR-Mamba, and AASIST as established detector backbones.

The repository contains the code developed for PGA-G, its pre-registered experimental protocols, per-repetition evaluation records, and executable numerical checks. These materials document the research contribution and support independent inspection of the experimental evidence. The journal manuscript and publication figures are maintained separately.

## Scientific scope

| Component | Implementation and evaluation |
| --- | --- |
| Selective adaptation | Prevalence gating, shortcut-consistency screening, and frozen/adapted branch selection |
| Operating points | Genuine-reference conformal calibration and independent-calibration design |
| Detectors | SSL-AASIST, XLSR-Mamba, and AASIST |
| Evaluation corpora | ASVspoof 2021 DF and ASVspoof 5 |
| Experimental evidence | Pre-registered stages, per-repetition records, source-grouped audits, and verification scripts |

## Quick start

Python **3.11+** is required for the pinned verification dependencies. The archived-evidence verification runs on CPU without downloading speech data or pretrained checkpoints.

```bash
python -m pip install -r requirements.txt
python scripts/verify_all.py
python scripts/check_release.py
python -m pytest -q
```

The verification suite checks recorded file hashes, recomputes stored confirmatory results and numerical summaries, and regenerates the source-grouped summary byte-identically. The numerical checks validate archived experimental records; independently rerunning detector inference requires access to the original external datasets and model checkpoints.

## Reproduction scope

| Reproduction task | Status |
| --- | --- |
| File-integrity checks and archived confirmatory numerical results | Automated and checked in CI |
| Seven condition-level evidence datasets from per-repetition records | Recomputable with the [evidence exporter](scripts/export_evidence_tables.py) |
| Source-grouped numerical summary | Byte-identical regeneration |
| Exact manuscript tables and publication figures | Not provided in this software repository |
| Detector inference and full experimental repetition | Requires external model code, checkpoints, audio and additional validation |

The [detailed reproduction scope](docs/REPRODUCTION_SCOPE.md) documents the known inference dependency gaps, acquisition references and provenance limitations. These distinctions are intentional: verification of released numerical evidence is not the same as rerunning speech-detector inference.

## Citation and licensing

Use [`CITATION.cff`](CITATION.cff) to cite the research software; the accompanying article should also be cited once published. The original software is distributed under the [MIT licence](LICENSE), whereas original released research records are licensed separately as described in [`docs/LICENSING.md`](docs/LICENSING.md). The repository does not grant rights to the upstream speech corpora or detector weights.

## Research components

| Directory | Purpose |
| --- | --- |
| [`pga_g/`](pga_g/) | Original adaptation framework, conformal calibration, detector interfaces and experimental modules |
| [`preregistration/`](preregistration/) | Fixed protocols for sequential experiments and design audits |
| [`results/`](results/) | Frozen per-repetition evaluations, outcomes, and machine-readable summaries |
| [`scripts/`](scripts/) | Offline integrity and numerical verification |
| [`tests/`](tests/) | Automated regression and research-code checks |
| [`docs/`](docs/) | Reproduction instructions, external resources and distribution scope |
| [`provenance/`](provenance/) | Research-record integrity manifest and provenance notes |

## Verification and provenance

The study reports both matched-reference and speaker-disjoint evaluations, including source-grouped and independently calibrated variants. Archived pre-registration records and the unmodified final v5 protocol hash document the recorded study sequence; hashes alone do not independently establish the original registration timestamps. Historical pilot naming is retained where it is part of a fixed protocol record; the maintained implementation is located at [`pga_g/pilot.py`](pga_g/pilot.py).

The released code and derived numerical records do not include the journal submission, publication figures, raw speech recordings, third-party model weights or unlicensed upstream implementations. See the [reproducibility guide](docs/REPRODUCIBILITY.md), [data and model sources](docs/DATA_AND_MODELS.md) and [record provenance](docs/RECORD_PROVENANCE.md).
