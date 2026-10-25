# Reproducibility scope and inference dependencies

The numerical verification suite is independent of detector checkpoints. Full detector inference is a separate, more demanding reproduction path.

| Reproduction level | Current status | Entry point or evidence |
| --- | --- | --- |
| SHA-256 integrity of released records | Verified by CI | `python scripts/verify_all.py` |
| Confirmatory quantities and reported aggregate values | Verified by CI | `scripts/verify_records.py` and `scripts/verify_summary.py` |
| Source-grouped numerical summary | Byte-identically reproducible | `scripts/regenerate_summary.py` |
| Seven condition-level evidence datasets | Deterministically recomputable from released CSV records | `python scripts/export_evidence_tables.py --output <directory>` |
| Exact typesetting of every manuscript table and figure | Not included | Requires manuscript-specific table builders and some additional material |
| Detector inference from original audio | Not self-contained in this release | Requires external code, licensed checkpoints, original audio and environment reconstruction |
| Full experimental rerun | Not independently demonstrated | Requires upstream resources, source-recording groups, protocol splits and compute |

## Detector sources and dependency boundary

- **AASIST:** original implementation at `https://github.com/clovaai/aasist`, revision `a04c9863f63d44471dde8a6abcb3b082b07cd1d1`; the archived research protocol identifies `AASIST.pth` with SHA-256 `51d2d9cf0738172f61e2a384ec50a54a55363240f67c971ed55a92435bc1a1c0`. The code expects an official model module at `pga_g/third_party/aasist_official.py`. That module and the checkpoint are not distributed here.
- **SSL-AASIST:** published upstream implementation at `https://github.com/TakHemlata/SSL_Anti-spoofing`. The local scorer expects a study-specific Hugging Face/fairseq weight-conversion adapter at `pga_g/aasist.py`; it is not included in the current package. The upstream model alone is not a drop-in substitute for this adapter.
- **XLSR-Mamba:** published code at `https://github.com/swagshaw/XLSR-Mamba`, commit `10ec73810d24091028a2aa5454c8dfd1a239e441`, with checkpoint distribution `https://huggingface.co/AustinXiao/XLSR-Mamba-DF`. `pga_g/mamba_pt.py` also imports functions from the missing `pga_g/aasist.py` adapter.

These missing imports are a known release-completeness limitation; installing numerical dependencies does **not** make the inference path executable. This document deliberately does not claim end-to-end reproduction. Before claiming it, provide the compatible internal adapters or a validated pinned reconstruction procedure, test strict checkpoint loading, and run a representative scored batch against recorded reference logits.

## Evidence generation

After installing `requirements.txt`, run:

```bash
python scripts/verify_all.py
python -m pytest -q
python scripts/export_evidence_tables.py --output generated-evidence
```

The seven generated CSVs are condition-level numerical summaries, **not** copies of manuscript tables, proofs of independently retrained detectors, or a replacement for the article's table-generation source. They deliberately preserve full output precision and link every aggregate to its input record path. Do not quote them as fully regenerated publisher tables.

## Recorded pre-registration timing

Protocol-object SHA-256 hashes attest to byte-level identity under their documented canonical serialization. The stored `written` and `fixed_on` fields describe the study's recorded chronology but are not independently timestamped by the later GitHub repository commit. Earlier externally timestamped evidence must be supplied separately if a claim of independently demonstrated prospective registration is required. The terminology-normalized initial protocol is not represented as an unchanged source file.

## Reproduction environment

The offline verification suite is tested with Python 3.11 and the pinned versions in `requirements.txt`. The included GitHub Actions workflow specifies the executable verification commands.
