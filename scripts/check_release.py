"""Fail on incomplete evidence-release metadata or checksum-manifest coverage.

This audits *repository packaging*, not independent detector inference.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_DIRS = {".git", ".pytest_cache", "__pycache__", ".venv", "venv", "build", "dist",
                 ".tox", "generated-evidence"}
EXCLUDED_FILE = "provenance/SHA256_MANIFEST.json"
REQUIRED = {"README.md", "CITATION.cff", "LICENSE", "pyproject.toml",
            "docs/REPRODUCTION_SCOPE.md", "scripts/verify_all.py",
            "scripts/export_evidence_tables.py"}
FORBIDDEN_EXTENSIONS = {".wav", ".flac", ".mp3", ".pth", ".pt", ".ckpt", ".safetensors"}
def audit() -> list[str]:
    errors = []
    present = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*")
               if p.is_file() and not any(parent in EXCLUDED_DIRS
                                          for parent in p.relative_to(ROOT).parts)}
    for name in sorted(REQUIRED - present):
        errors.append(f"required release file missing: {name}")
    for name in sorted(present):
        if Path(name).suffix.lower() in FORBIDDEN_EXTENSIONS:
            errors.append(f"third-party data/weights should not be bundled: {name}")
    mf = ROOT / EXCLUDED_FILE
    if not mf.is_file():
        return errors + ["missing provenance/SHA256_MANIFEST.json"]
    manifest = json.loads(mf.read_text(encoding="utf-8"))
    listed = [row["path"] for row in manifest]
    if len(listed) != len(set(listed)):
        errors.append("duplicate paths in SHA-256 manifest")
    extra = set(listed) - (present - {EXCLUDED_FILE})
    missing = (present - {EXCLUDED_FILE}) - set(listed)
    errors += [f"manifest lists absent file: {p}" for p in sorted(extra)]
    errors += [f"manifest omits release file: {p}" for p in sorted(missing)]
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    if 'requires-python = ">=3.11"' not in pyproject:
        errors.append("pyproject must require Python >=3.11 for pinned scikit-learn 1.8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    if "docs/REPRODUCTION_SCOPE.md" not in readme:
        errors.append("README does not link the honest reproduction-scope documentation")
    return errors

if __name__ == "__main__":
    problems = audit()
    for problem in problems:
        print("ERROR:", problem)
    if problems:
        raise SystemExit(f"Evidence release audit failed ({len(problems)} issue(s))")
    print("Evidence release metadata and manifest coverage: PASS")
    print("Scope: archived numerical evidence; detector inference is not self-contained")
