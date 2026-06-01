"""Cross-session artifact persistence for the split notebooks.

Each phase notebook (01_discovery → 02_benchmark → 03_scale → 04_sycophancy)
runs in its own Colab Enterprise runtime, which has a fresh ``/content``. To
hand artifacts (the discovery checkpoint, the H6 jsonls, comparison.csv …)
from one phase to the next, every notebook **restores** ``results/`` at the
top and **mirrors** it back at the bottom, through a shared backend.

Backends (auto-detected, override with ``DP_PERSIST``):

  * ``gcs``   — a Google Cloud Storage bucket (``GCS_BUCKET=gs://my-bucket``).
                Best on Colab Enterprise / Vertex: gsutil is pre-installed and
                the runtime service account already has bucket access.
  * ``drive`` — Google Drive mounted at ``/content/drive`` (free Colab). Set
                up by the notebook calling ``google.colab.drive.mount`` first.
  * ``local`` — no shared backend; artifacts live only in this runtime. Fine
                for a single-session monolith run, but the split phases won't
                see each other's outputs — the notebook warns loudly.

Only the *pure* pieces (backend detection, command building, URI
normalization) live here so they're unit-testable without a cloud account;
the notebook does the actual ``subprocess`` call.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Mapping, Optional


VALID_BACKENDS = ("gcs", "drive", "local")
DEFAULT_DRIVE_ROOT = "/content/drive/MyDrive/diagnosticpercept"


def detect_backend(env: Mapping[str, str], drive_available: bool) -> str:
    """Decide the persistence backend.

    Precedence: explicit ``DP_PERSIST`` → ``GCS_BUCKET`` set → mounted Drive
    → ``local``. ``drive_available`` is whether ``/content/drive`` is mounted.
    """
    override = env.get("DP_PERSIST", "").strip().lower()
    if override:
        if override not in VALID_BACKENDS:
            raise ValueError(
                f"DP_PERSIST={override!r} invalid; choose one of {VALID_BACKENDS}."
            )
        return override
    if env.get("GCS_BUCKET", "").strip():
        return "gcs"
    if drive_available:
        return "drive"
    return "local"


def gcs_uri(bucket: str, subpath: str = "results") -> str:
    """Normalize ``bucket`` (with or without ``gs://``) to a full URI."""
    b = bucket.strip()
    if b.startswith("gs://"):
        b = b[len("gs://"):]
    b = b.strip("/")
    sub = subpath.strip("/")
    return f"gs://{b}/{sub}"


def drive_dir(root: str = DEFAULT_DRIVE_ROOT, subpath: str = "results") -> str:
    return str(Path(root) / subpath)


def build_sync_cmd(src: str, dst: str, backend: str) -> List[str]:
    """Argv to sync ``src`` → ``dst`` for ``backend``.

    - gcs: ``gsutil -m rsync -r src dst`` (src or dst may be a gs:// URI).
    - drive/local: ``rsync -a src/ dst`` (trailing slash on src copies
      *contents*, matching gsutil rsync semantics).
    """
    if backend == "gcs":
        return ["gsutil", "-m", "rsync", "-r", src, dst]
    if backend in ("drive", "local"):
        s = src.rstrip("/") + "/"
        return ["rsync", "-a", s, dst]
    raise ValueError(f"unknown backend: {backend!r}")


def remote_location(
    backend: str,
    *,
    bucket: Optional[str] = None,
    drive_root: str = DEFAULT_DRIVE_ROOT,
    subpath: str = "results",
) -> Optional[str]:
    """Where the shared copy lives, or ``None`` for ``local`` (no remote)."""
    if backend == "gcs":
        if not bucket:
            raise ValueError("gcs backend requires a bucket (set GCS_BUCKET).")
        return gcs_uri(bucket, subpath)
    if backend == "drive":
        return drive_dir(drive_root, subpath)
    return None
