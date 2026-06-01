"""Tests for src/persist.py — backend detection + sync command building."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ---- detect_backend ----


def test_detect_explicit_override():
    from src.persist import detect_backend
    assert detect_backend({"DP_PERSIST": "gcs"}, drive_available=False) == "gcs"
    assert detect_backend({"DP_PERSIST": "local"}, drive_available=True) == "local"


def test_detect_invalid_override_raises():
    from src.persist import detect_backend
    with pytest.raises(ValueError, match="invalid"):
        detect_backend({"DP_PERSIST": "dropbox"}, drive_available=False)


def test_detect_gcs_from_bucket_env():
    from src.persist import detect_backend
    assert detect_backend({"GCS_BUCKET": "gs://my-bucket"}, drive_available=True) == "gcs"


def test_detect_drive_when_mounted():
    from src.persist import detect_backend
    assert detect_backend({}, drive_available=True) == "drive"


def test_detect_local_default():
    from src.persist import detect_backend
    assert detect_backend({}, drive_available=False) == "local"


# ---- gcs_uri ----


@pytest.mark.parametrize("bucket,expected", [
    ("gs://my-bucket", "gs://my-bucket/results"),
    ("my-bucket", "gs://my-bucket/results"),
    ("gs://my-bucket/", "gs://my-bucket/results"),
    ("my-bucket/sub", "gs://my-bucket/sub/results"),
])
def test_gcs_uri_normalization(bucket, expected):
    from src.persist import gcs_uri
    assert gcs_uri(bucket) == expected


def test_gcs_uri_custom_subpath():
    from src.persist import gcs_uri
    assert gcs_uri("my-bucket", subpath="results/h6") == "gs://my-bucket/results/h6"


# ---- build_sync_cmd ----


def test_build_sync_cmd_gcs():
    from src.persist import build_sync_cmd
    cmd = build_sync_cmd("/content/results", "gs://b/results", "gcs")
    assert cmd == ["gsutil", "-m", "rsync", "-r", "/content/results", "gs://b/results"]


def test_build_sync_cmd_drive_adds_trailing_slash():
    from src.persist import build_sync_cmd
    cmd = build_sync_cmd("/content/results", "/content/drive/MyDrive/x/results", "drive")
    assert cmd == ["rsync", "-a", "/content/results/", "/content/drive/MyDrive/x/results"]


def test_build_sync_cmd_unknown_raises():
    from src.persist import build_sync_cmd
    with pytest.raises(ValueError, match="unknown backend"):
        build_sync_cmd("a", "b", "ftp")


# ---- remote_location ----


def test_remote_location_gcs():
    from src.persist import remote_location
    assert remote_location("gcs", bucket="my-bucket") == "gs://my-bucket/results"


def test_remote_location_gcs_without_bucket_raises():
    from src.persist import remote_location
    with pytest.raises(ValueError, match="requires a bucket"):
        remote_location("gcs")


def test_remote_location_drive():
    from src.persist import remote_location
    assert remote_location("drive").endswith("/diagnosticpercept/results")


def test_remote_location_local_is_none():
    from src.persist import remote_location
    assert remote_location("local") is None


# ---- PeriodicMirror ----


def test_periodic_mirror_syncs_then_final(monkeypatch):
    """Thread fires at least once on a short interval, plus a final sync on stop."""
    import threading
    import time as _time

    from src.persist import PeriodicMirror

    calls = []
    gate = threading.Event()

    def fake_sync(local, remote, backend):
        calls.append((local, remote, backend))
        gate.set()

    pm = PeriodicMirror("/local", "gs://b/results", "gcs",
                        interval=0.05, sync_fn=fake_sync)
    pm.start()
    # Wait for at least one periodic tick (bounded so the test can't hang).
    assert gate.wait(timeout=5.0), "periodic sync never fired"
    n_periodic = len(calls)
    pm.stop(final=True)
    # stop(final=True) adds exactly one more sync.
    assert len(calls) == n_periodic + 1
    assert calls[-1] == ("/local", "gs://b/results", "gcs")


def test_periodic_mirror_stop_without_start_does_final_only():
    from src.persist import PeriodicMirror
    calls = []
    pm = PeriodicMirror("/l", "/r", "drive", interval=999,
                        sync_fn=lambda *a: calls.append(a))
    pm.stop(final=True)
    assert len(calls) == 1


def test_periodic_mirror_survives_sync_exception(capsys):
    from src.persist import PeriodicMirror

    def boom(*a):
        raise RuntimeError("network blip")

    pm = PeriodicMirror("/l", "/r", "gcs", interval=999, sync_fn=boom)
    pm._sync_once()   # must not raise
    out = capsys.readouterr().out
    assert "sync failed" in out


def test_periodic_mirror_context_manager(monkeypatch):
    from src.persist import PeriodicMirror
    calls = []
    with PeriodicMirror("/l", "/r", "local", interval=999,
                        sync_fn=lambda *a: calls.append(a)):
        pass
    # __exit__ → stop(final=True) → one sync.
    assert len(calls) == 1
