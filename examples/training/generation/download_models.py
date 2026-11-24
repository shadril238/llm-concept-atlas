"""Download immutable public files, verify every byte, never load remote Python code."""

import hashlib
import json
import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[3]
SPECS = json.loads((Path(__file__).parent / "models.json").read_text())


def download(name):
    spec = SPECS[name]
    target = ROOT / ".cache/e8" / name.replace("/", "--") / spec["revision"]
    target.mkdir(parents=True, exist_ok=True)
    for f in spec["files"]:
        p = target / f["name"]
        if not p.exists():
            cached = hf_hub_download(
                name,
                f["name"],
                revision=spec["revision"],
                token=False,
                cache_dir=ROOT / ".cache/e8/hub",
                local_files_only=os.getenv("HF_HUB_OFFLINE") == "1",
            )
            shutil.copy2(cached, p)
        if (
            p.stat().st_size != f["size"]
            or hashlib.sha256(p.read_bytes()).hexdigest() != f["sha256"]
        ):
            raise ValueError("Model file checksum mismatch: " + f["name"])
    return str(target)
