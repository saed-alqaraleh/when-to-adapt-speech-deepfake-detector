"""Run the offline, CPU-based reproducibility verification suite."""
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
for name in ("verify_integrity.py", "verify_records.py", "verify_summary.py", "regenerate_summary.py"):
    print(f"\n== {name} ==", flush=True)
    subprocess.run([sys.executable, str(root / "scripts" / name)], cwd=root, check=True)
