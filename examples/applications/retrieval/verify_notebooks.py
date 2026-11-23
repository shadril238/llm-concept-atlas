"""Restart a fresh actual Jupyter kernel for each notebook, then execute every cell."""

from pathlib import Path
import argparse
import datetime
import platform
import hashlib
import json
import os
import sys
import tempfile

import nbformat
from nbclient import NotebookClient

PROJECT = Path(__file__).resolve().parent
ROOT = PROJECT.parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="Permit the first pinned public GGUF download",
    )
    parser.add_argument("--id", action="append", default=[])
    args = parser.parse_args()
    output = ROOT / "examples/artifacts/e6-notebooks"
    output.mkdir(parents=True, exist_ok=True)
    records = []
    # Task-local kernel registration avoids modifying user Jupyter settings.
    with tempfile.TemporaryDirectory(prefix="atlas-e6-jupyter-") as temporary:
        directory = Path(temporary)
        kernel = directory / "kernels/python3"
        kernel.mkdir(parents=True)
        (kernel / "kernel.json").write_text(
            json.dumps(
                {
                    "argv": [
                        sys.executable,
                        "-m",
                        "ipykernel_launcher",
                        "-f",
                        "{connection_file}",
                    ],
                    "display_name": "Python 3.11 (atlas retrieval)",
                    "language": "python",
                }
            )
        )
        original = os.environ.get("JUPYTER_PATH")
        os.environ["JUPYTER_PATH"] = str(directory)
        os.environ["JUPYTER_RUNTIME_DIR"] = str(directory / "runtime")
        os.environ["IPYTHONDIR"] = str(directory / "ipython")
        os.environ["HF_HUB_OFFLINE"] = "0" if args.allow_network else "1"
        try:
            for path in sorted((PROJECT / "notebooks").glob("*.ipynb")):
                if (args.id and path.stem not in args.id) or (
                    not args.id and path.stem == "blip2-captioning"
                ):
                    continue
                notebook = nbformat.read(path, as_version=4)
                # Each client creates and shuts down a new kernel. No preexisting state.
                NotebookClient(
                    notebook,
                    timeout=600,
                    kernel_name="python3",
                    resources={"metadata": {"path": str(PROJECT)}},
                ).execute()
                nbformat.write(notebook, output / path.name)
                count = sum(cell.cell_type == "code" for cell in notebook.cells)
                records.append(
                    {
                        "path": str(path.relative_to(ROOT)),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "method": "Fresh actual Jupyter kernel; all cells executed in order via nbclient",
                        "codeCells": count,
                        "status": "passed",
                    }
                )
                print(path.name, count, "code cells passed", flush=True)
        finally:
            if original is None:
                os.environ.pop("JUPYTER_PATH", None)
            else:
                os.environ["JUPYTER_PATH"] = original
    (
        PROJECT
        / (
            "notebook-execution-blip2.json"
            if args.id == ["blip2-captioning"]
            else "notebook-execution-local.json"
        )
    ).write_text(
        json.dumps(
            {
                "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "platform": platform.platform(),
                "lockSha256": hashlib.sha256(
                    (PROJECT / "uv.lock").read_bytes()
                ).hexdigest(),
                "verifierSha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "python": sys.version.split()[0],
                "records": records,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
