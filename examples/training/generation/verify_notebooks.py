"""Restart a fresh actual Jupyter kernel for each notebook, then execute every cell."""

import argparse
import datetime
import hashlib
import json
import os
import platform
import sys
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient

PROJECT = Path(__file__).resolve().parent
ROOT = PROJECT.parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="Permit the first pinned public model download",
    )
    parser.add_argument("--id", action="append", default=[])
    args = parser.parse_args()
    output = ROOT / "examples/artifacts/e8-notebooks"
    output.mkdir(parents=True, exist_ok=True)
    records = []
    # Task-local kernel registration avoids modifying user Jupyter settings.
    with tempfile.TemporaryDirectory(prefix="atlas-e8-jupyter-") as temporary:
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
                    "display_name": "Python 3.11 (atlas generation)",
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
            for path in sorted(
                (PROJECT / "notebooks").glob("*.ipynb"),
                key=lambda p: [
                    "instruction-adaptation",
                    "preference-adaptation",
                    "generation-evaluation",
                ].index(p.stem),
            ):
                if args.id and path.stem not in args.id:
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
        / ("notebook-execution-selected.json" if args.id else "notebook-execution.json")
    ).write_text(
        json.dumps(
            {
                "date": datetime.datetime.now(datetime.UTC).isoformat(),
                "platform": platform.platform(),
                "lockSha256": hashlib.sha256(
                    (PROJECT / "uv.lock").read_bytes()
                ).hexdigest(),
                "verifierSha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "python": sys.version.split()[0],
                "sourceHashes": {
                    str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(
                        [PROJECT / "src/atlas_generation" / name for name in ["common.py", "sft.py", "preferences.py", "evaluate.py"]]
                        + [
                            PROJECT / "download_models.py",
                            PROJECT / "models.json",
                            ROOT / "examples/data/fixtures/instruction-dolly.json",
                            ROOT / "examples/data/fixtures/instruction-contract.json",
                            ROOT / "examples/data/fixtures/generation-preferences.json",
                            PROJECT / "notebook_views.py",
                            ROOT
                            / "examples/foundations/src/atlas_foundations/e8/contracts.py",
                        ]
                    )
                },
                "records": records,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
