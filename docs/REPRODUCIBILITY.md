# Reproducibility

## Offline evidence checks

`python scripts/verify_all.py` validates frozen-file checksums, reconstructs per-repetition hypothesis quantities, independently verifies the stored numerical summary and regenerates the source-grouped summary byte-identically. It neither downloads audio nor trains a detector. The focused release tests run with `python -m pytest -q`.

The archived rows are the output of fixed pre-registered experiments. Computing the same derived quantities from the included rows is distinct from repeating the original detector inference from raw audio. The latter requires licensed third-party datasets and pretrained model checkpoints. Detector-dependent test cases may require the original upstream software and a GPU.

## Evaluation structure

- The initial prevalence gate uses a frozen-score mixture-proportion estimate.
- The extension adds a consistency guard to screen reference speaker shortcuts.
- Generalisation spans three detectors and two speech deepfake evaluation corpora.
- The later source-aware audit compares original and grouped protocols, and separates branch selection from final calibration for the independent-calibration design.

The experiment folders under `results/` retain the original study grouping. Stage names and detector-specific conditions are preserved. Output labels using private pilot terminology have been normalized consistently, without altering any numeric observation.

## Boundaries

This research package does not redistribute original speech, benchmark packages, pretrained detector checkpoints or upstream model implementations. Certain source-recording assignment files were withheld pending redistribution review. The historical source code contains software hooks for external detector components; obtain those from their cited upstream distributions.

The exploratory pilot command-line implementation is available as `python -m pga_g.pilot --help`. Its original historical identifier is preserved in frozen protocol text.
