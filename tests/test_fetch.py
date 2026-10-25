"""Cache/copy/extract logic of pga_g.fetch with local files (no network). Downloads are not tested."""
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pga_g import fetch  # noqa: E402


def _make_zip(path: Path):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("LA/ASVspoof2019_LA_eval/flac/a.flac", b"x" * 10)
        z.writestr("LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt", "s a - - bonafide\n")
        z.writestr("LA/ASVspoof2019_LA_train/flac/b.flac", b"y" * 10)


def test_cache_copy_and_member_extraction(tmp_path, monkeypatch):
    src = dict(fetch.SOURCES["la19"], min_bytes=10)
    monkeypatch.setitem(fetch.SOURCES, "la19", src)
    drive, local = tmp_path / "drive", tmp_path / "local"
    drive.mkdir()
    _make_zip(drive / "LA.zip")

    def no_download(*a, **k):
        raise AssertionError("must not download when a valid Drive copy exists")
    monkeypatch.setattr(fetch, "_download_url", no_download)

    p = fetch.ensure("la19", local, drive, cache_on_drive=True)
    assert p == local / "LA.zip" and p.exists()
    assert fetch.ensure("la19", local, drive, cache_on_drive=True) == p      # local reused
    out = fetch.extract("la19", p, tmp_path / "data")
    assert (out / "LA/ASVspoof2019_LA_eval/flac/a.flac").exists()
    assert (out / "LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt").exists()
    assert not (out / "LA/ASVspoof2019_LA_train").exists()                  # only needed members
    marker = (out / "_EXTRACTED.json").stat().st_mtime
    fetch.extract("la19", p, tmp_path / "data")                              # idempotent
    assert (out / "_EXTRACTED.json").stat().st_mtime == marker


def test_truncated_zip_is_rejected(tmp_path, monkeypatch):
    src = dict(fetch.SOURCES["la19"], min_bytes=10)
    monkeypatch.setitem(fetch.SOURCES, "la19", src)
    z = tmp_path / "LA.zip"
    _make_zip(z)
    data = z.read_bytes()
    z.write_bytes(data[: len(data) - 40])                                    # cut the central directory
    assert not fetch._valid(z, src)


def test_la19_mirror_materialisation(tmp_path):
    """Fake mirror in the exact schema (path, audio{bytes,path}, label 0=bonafide, notes JSON)."""
    import io
    import json as _json
    import numpy as np
    import pytest
    pytest.importorskip("pyarrow")
    import pyarrow as pa
    import pyarrow.parquet as pq
    import soundfile as sf
    from pga_g import data as D

    mirror = tmp_path / "mirror"
    (mirror / "data").mkdir(parents=True)
    (mirror / "protocols").mkdir()
    rng = np.random.default_rng(0)
    proto, n_per, n_shards = [], 3, 3
    orig = {}
    for s in range(n_shards):
        rows = {"path": [], "audio": [], "label": [], "notes": []}
        for i in range(n_per):
            utt = f"LA_E_{1000000 + s * 10 + i}"
            buf = io.BytesIO()
            x = (rng.standard_normal(16000) * 3000).astype(np.int16)
            sf.write(buf, x, 16000, format="WAV" if (s, i) == (1, 1) else "FLAC", subtype="PCM_16")
            lab = (s + i) % 2
            orig[utt] = x
            rows["path"].append(f"{utt}.flac")
            rows["audio"].append({"bytes": buf.getvalue(), "path": f"{utt}.flac"})
            rows["label"].append(lab)
            rows["notes"].append(_json.dumps({"utterance_id": utt, "speaker_id": "LA_0001", "subset": "eval"}))
            proto.append(f"LA_0001 {utt} - {'-' if lab == 0 else 'A07'} {'bonafide' if lab == 0 else 'spoof'}")
        pq.write_table(pa.table(rows), mirror / "data" / f"test-{s:05d}-of-{n_shards:05d}.parquet")
    (mirror / "protocols" / fetch.PROTO_NAME).write_text("\n".join(proto) + "\n")

    old = dict(fetch.LA19_MIRROR)
    fetch.LA19_MIRROR["expected_rows"] = n_per * n_shards
    try:
        dest = fetch.materialise_la19(mirror, "abc123", tmp_path / "la19")
        rep = _json.loads((dest / "_EXTRACTED.json").read_text())
        assert rep["rows"] == 9 and rep["reencoded"] == 1
        man = D.build_la19_eval_manifest(dest)
        assert len(man) == 9 and (man.label == 1).sum() == sum(1 for l in proto if l.endswith("bonafide"))
        for utt, x in orig.items():                       # audio round-trips exactly
            y, sr = sf.read(dest / "LA/ASVspoof2019_LA_eval/flac" / f"{utt}.flac", dtype="int16")
            assert sr == 16000 and np.array_equal(x, y)
        # a label disagreement with the protocol must be caught
        bad = (mirror / "protocols" / fetch.PROTO_NAME).read_text().replace("bonafide", "spoof", 1)
        (mirror / "protocols" / fetch.PROTO_NAME).write_text(bad)
        import pytest
        with pytest.raises(RuntimeError):
            fetch.materialise_la19(mirror, "def456", tmp_path / "la19b")
    finally:
        fetch.LA19_MIRROR.clear()
        fetch.LA19_MIRROR.update(old)
