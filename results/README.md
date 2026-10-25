# Archived numerical evidence

Every `results.csv` is an experiment-cell record with the original numeric values retained. Directories group results by protocol stage, corpus, detector, source-reference mode, deployment share and repetition. Associated `config.json` files retain the setup where releasable; the verdict JSON files preserve the hypothesis-decision summaries. `results/verification/numbers_v5.json` supports independent numerical consistency checks.

Run `python scripts/verify_all.py` to recompute the confirmatory quantities and verify the release manifest. This verification is independent of GPU inference and does not imply that excluded checkpoint files are packaged.

Dataset-specific trial source maps, raw audio and personal workspace logs are not part of the bundle. Names of early-stage method variants were normalized consistently in source code, saved output labels and verification scripts; numerical fields were not changed.
