# Record provenance and naming

Frozen experiment records are retained at their original precision. Historical private working identifiers have been normalized in the distributed software and selected early-stage result labels. This change is textual, not a re-evaluation or new experimental run. Original source records and their hash inventory are retained separately from this distributable repository.

The last split-audit pre-registration and its SHA-256 are preserved without modification. Its canonical protocol-object digest is recorded in `provenance/record-scope.json`. The initial prevalence-test pre-registration includes a terminology-normalized copy; it is explicitly identified as a derivative, **not** a newly issued or unmodified original pre-registration.

All numerical verification scripts are based on the stored per-repetition files and exit with an error if the verified quantities disagree. The checksum manifest authenticates this distributed copy only and should not be confused with the checksum of an earlier private source archive.

The former root-level pilot script is organized under `pga_g/pilot.py`; dependent imports and the pilot runner have been updated without changing scoring, statistical routines or stored results. The licensing-scope document is now under `docs/LICENSING.md`.
