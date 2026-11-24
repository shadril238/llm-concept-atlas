"""Pinned public downloads, explicit file completeness and weight SHA-256 checks."""

import hashlib
import json
import os
from pathlib import Path
import shutil
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[3]
MODELS = json.loads((Path(__file__).parent / "models.json").read_text())


def download(name):
    spec = MODELS[name]
    target = ROOT / ".cache/e7" / name.replace("/", "--") / spec["revision"]
    target.mkdir(parents=True, exist_ok=True)
    for f in spec["files"]:
        p = target / f["name"]
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.is_file():
            cached = (
                Path.home()
                / ".cache/huggingface/hub"
                / ("models--" + name.replace("/", "--"))
                / "snapshots"
                / spec["revision"]
                / f["name"]
            )
            if cached.is_file():
                shutil.copy2(cached, p)
            else:
                path = hf_hub_download(
                    name,
                    f["name"],
                    revision=spec["revision"],
                    cache_dir=ROOT / ".cache/e7/hub",
                    token=False,
                    local_files_only=os.environ.get("HF_HUB_OFFLINE") == "1",
                )
                shutil.copy2(path, p)
        if f.get("size") is not None and p.stat().st_size != f["size"]:
            raise ValueError("Model file size mismatch: " + f["name"])
        if f.get("sha256"):
            h = hashlib.sha256()
            with p.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
                    h.update(chunk)
            if h.hexdigest() != f["sha256"]:
                raise ValueError("Model checksum mismatch: " + f["name"])
    return str(target)
