"""Check SHA-256 digests of every bundled research artifact."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "provenance" / "SHA256_MANIFEST.json").read_text(encoding="utf8"))
errors = []
for item in manifest:
    file = root / item["path"]
    if not file.is_file():
        errors.append(f"Missing: {item['path']}")
        continue
    h = hashlib.sha256(file.read_bytes()).hexdigest()
    if h != item["sha256"]:
        errors.append(f"Checksum mismatch: {item['path']}")
print(f"Integrity: {len(manifest) - len(errors)}/{len(manifest)} files")
if errors:
    raise SystemExit("\n".join(errors))
