"""Download only pinned public safetensors and tokenizer/config files; never model code."""

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from huggingface_hub import snapshot_download

PROJECT = Path(__file__).resolve().parent
ROOT = PROJECT.parents[2]


def download(name):
    spec = json.loads((PROJECT / "models.json").read_text())[name]
    path = Path(
        snapshot_download(
            name,
            revision=spec["revision"],
            token=False,
            cache_dir=ROOT / ".cache/e6/hub",
            allow_patterns=["*.json", "*.txt", "*.model", "*.safetensors"],
            max_workers=1,
            local_files_only=os.getenv("HF_HUB_OFFLINE") == "1",
        )
    )
    for item in spec["files"]:
        file = path / item["name"]
        if not file.exists():
            raise FileNotFoundError(
                "Pinned snapshot is incomplete; finish the public download before offline execution: "
                + item["name"]
            )
        if item["sha256"]:
            h = hashlib.sha256()
            with file.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    h.update(block)
            if h.hexdigest() != item["sha256"]:
                raise ValueError("Artifact checksum mismatch: " + item["name"])
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+")
    args = parser.parse_args()
    for name in args.models:
        path = download(name)
        print(name, "pinned snapshot ready", flush=True)
