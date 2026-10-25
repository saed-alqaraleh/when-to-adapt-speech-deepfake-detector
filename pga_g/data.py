"""Manifests and audio loading for In-the-Wild and ASVspoof 2019 LA eval."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import soundfile as sf

SAMPLE_RATE = 16000
CUT = 64600

MANIFEST_COLUMNS = ["idx", "utt_id", "relpath", "speaker", "label", "attack"]
# label: 1 = bona fide, 0 = spoof (same convention as the official repo)

ITW_EXPECTED = {"n": 31779, "bona": 19963, "spoof": 11816}   # Zhou & Wang 2026, Sec. 4.1
LA19_EVAL_EXPECTED = {"n": 71237, "bona": 7355, "spoof": 63882}  # ASVspoof 2019 LA eval protocol


def official_pad(x: np.ndarray, max_len: int = CUT) -> np.ndarray:
    """Exactly the official data_utils_SSL.pad: crop the first max_len samples, or tile."""
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    return np.tile(x, (1, num_repeats))[:, :max_len][0]


def load_audio(path: str) -> tuple[np.ndarray, int, int]:
    """float32 mono at 16 kHz (librosa.load(sr=16000) equivalent: channel mean + soxr HQ)."""
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
    n_orig = x.shape[0]
    if sr != SAMPLE_RATE:
        import soxr
        x = soxr.resample(x, sr, SAMPLE_RATE, quality="HQ").astype(np.float32)
    if x.size == 0:
        raise ValueError(f"empty audio: {path}")
    return np.ascontiguousarray(x), int(sr), int(n_orig)


def load_audio_bytes(blob: bytes) -> tuple[np.ndarray, int, int]:
    """Same as load_audio, from in-memory encoded audio (e.g. FLAC bytes in a parquet cell)."""
    import io
    x, sr = sf.read(io.BytesIO(blob), dtype="float32", always_2d=True)
    x = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
    n_orig = x.shape[0]
    if sr != SAMPLE_RATE:
        import soxr
        x = soxr.resample(x, sr, SAMPLE_RATE, quality="HQ").astype(np.float32)
    if x.size == 0:
        raise ValueError("empty audio blob")
    return np.ascontiguousarray(x), int(sr), int(n_orig)


def trim_silence(x: np.ndarray, top_db: float, min_keep: int = 1600) -> tuple[int, int]:
    """Leading/trailing silence removal exactly as Mueller et al. (2021) measured it:
    librosa.effects.trim with a 40 dB threshold (defaults: 2048-sample frames, hop 512).
    Returns [start, end) in samples; the whole clip if fewer than min_keep samples would remain."""
    import librosa
    _, (a, b) = librosa.effects.trim(x, top_db=top_db)
    a, b = int(a), int(b)
    if b - a < min_keep:
        return 0, int(x.size)
    return a, b


class BytesAudioDataset:
    """Decodes encoded audio blobs held in memory (shared with DataLoader workers by fork).
    With trim_db, leading/trailing silence is removed before the official crop/tile and two
    extra fields are returned: leading silence and kept length (samples at 16 kHz)."""

    def __init__(self, blobs, trim_db: Optional[float] = None, cut: int = CUT):
        self.blobs = blobs
        self.trim_db = trim_db
        self.cut = cut

    def __len__(self):
        return len(self.blobs)

    def __getitem__(self, i):
        x, sr, n = load_audio_bytes(self.blobs[i])
        if not self.trim_db:
            return official_pad(x, self.cut), int(i), sr, n
        a, b = trim_silence(x, self.trim_db)
        return official_pad(x[a:b], self.cut), int(i), sr, n, a, b - a


# ------------------------------------------------------------------------- manifests
def _norm_label(v: str) -> Optional[int]:
    s = str(v).strip().lower().replace("_", "-").replace(" ", "-")
    if s in {"bona-fide", "bonafide", "bona", "genuine", "real"}:
        return 1
    if s in {"spoof", "fake", "deepfake", "spoofed"}:
        return 0
    return None


def find_one(root: Path, name: str) -> Path:
    hits = sorted(p for p in root.rglob(name) if not p.name.startswith("._"))
    if not hits:
        raise FileNotFoundError(f"{name} not found under {root}")
    if len(hits) > 1:
        hits.sort(key=lambda p: len(p.parts))
    return hits[0]


def build_itw_manifest(root: Path) -> pd.DataFrame:
    """Parse the In-the-Wild meta.csv without assuming its column names: the file column
    is the one whose values end in an audio extension, the label column the one whose
    values normalise to bona fide/spoof, the speaker column the remaining text column."""
    meta = find_one(root, "meta.csv")
    df = pd.read_csv(meta, dtype=str, keep_default_na=False)
    cols = list(df.columns)
    file_col = next((c for c in cols if df[c].str.lower().str.endswith((".wav", ".flac")).mean() > 0.99), None)
    label_col = next((c for c in cols if df[c].map(_norm_label).notna().mean() > 0.99), None)
    if file_col is None or label_col is None:
        raise ValueError(f"cannot identify file/label columns in {meta}; columns={cols}; head=\n{df.head()}")
    rest = [c for c in cols if c not in (file_col, label_col)]
    spk_col = max(rest, key=lambda c: df[c].nunique()) if rest else None
    audio_dir = meta.parent
    rel = df[file_col].map(lambda f: str((audio_dir / f).relative_to(root)))
    missing = [r for r in rel.iloc[:50] if not (root / r).exists()]
    if missing:
        raise FileNotFoundError(f"audio files listed in meta.csv not found, e.g. {missing[:3]}")
    out = pd.DataFrame({
        "utt_id": df[file_col].map(lambda f: Path(f).stem),
        "relpath": rel,
        "speaker": df[spk_col] if spk_col else "unknown",
        "label": df[label_col].map(_norm_label).astype(int),
        "attack": "-",
    })
    return _finalise(out)


def build_la19_eval_manifest(root: Path) -> pd.DataFrame:
    proto = find_one(root, "ASVspoof2019.LA.cm.eval.trl.txt")
    flac_dir = find_one(root, "ASVspoof2019_LA_eval") / "flac"
    rows = []
    with open(proto) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 5:
                continue
            spk, utt, _, attack, key = parts[:5]
            rows.append((utt, str((flac_dir / f"{utt}.flac").relative_to(root)), spk,
                         1 if key == "bonafide" else 0, attack))
    out = pd.DataFrame(rows, columns=["utt_id", "relpath", "speaker", "label", "attack"])
    missing = [r for r in out.relpath.iloc[:50] if not (root / r).exists()]
    if missing:
        raise FileNotFoundError(f"eval flac files not found, e.g. {missing[:3]}")
    return _finalise(out)


def _finalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("relpath", kind="mergesort").reset_index(drop=True)  # deterministic order
    if df["utt_id"].duplicated().any():
        raise ValueError("duplicate utt_id in manifest")
    df.insert(0, "idx", np.arange(len(df), dtype=np.int64))
    return df[MANIFEST_COLUMNS]


def check_expected(df: pd.DataFrame, expected: dict) -> dict:
    got = {"n": int(len(df)), "bona": int((df.label == 1).sum()), "spoof": int((df.label == 0).sum())}
    return {"expected": expected, "got": got, "match": got == expected}


def write_manifest(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, quoting=csv.QUOTE_MINIMAL)
    tmp.replace(path)


# ------------------------------------------------------------------------- torch dataset
class AudioDataset:
    """Returns (padded waveform float32[CUT], idx, original sample rate, original length)."""

    def __init__(self, root: Path, relpaths, idxs):
        self.root = str(root)
        self.relpaths = list(relpaths)
        self.idxs = np.asarray(idxs, dtype=np.int64)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        x, sr, n = load_audio(f"{self.root}/{self.relpaths[i]}")
        return official_pad(x), int(self.idxs[i]), sr, n


def collate(batch):
    import torch
    wav = torch.from_numpy(np.stack([b[0] for b in batch]))
    idx = torch.tensor([b[1] for b in batch], dtype=torch.int64)
    sr = torch.tensor([b[2] for b in batch], dtype=torch.int32)
    n = torch.tensor([b[3] for b in batch], dtype=torch.int64)
    extra = tuple(torch.tensor([b[k] for b in batch], dtype=torch.int64) for k in range(4, len(batch[0])))
    return (wav, idx, sr, n) + extra
