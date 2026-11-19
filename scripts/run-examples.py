#!/usr/bin/env python3
"""Execute only explicitly selected model labs; default is the offline suite."""

import argparse
import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

from example_tools import ROOT, contract_digest, digest, load, validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", action="append", default=[])
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--record",
        action="store_true",
        help="Write run records; successful records become manifest evidence",
    )
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if args.allow_network and args.offline:
        parser.error("Choose network or offline mode")
    manifest = load()
    errors = validate(manifest, check_records=False, check_generated=False)
    if errors:
        raise SystemExit("\n".join(errors))
    entries = manifest["examples"]
    if args.list:
        for entry in entries:
            print(entry["id"], entry["environment"], entry["network"])
        return
    unknown = set(args.id) - {e["id"] for e in entries}
    if unknown:
        parser.error("Unknown examples: " + ",".join(sorted(unknown)))
    selected = (
        [e for e in entries if e["id"] in args.id]
        if args.id
        else [e for e in entries if e["environment"] == "foundations"]
    )
    failures = []
    for entry in selected:
        if entry["environment"] == "legacy-unverified":
            raise SystemExit(
                entry["id"] + ": unpinned legacy sketch; migrate before execution"
            )
        if entry["network"] != "offline" and not (args.allow_network or args.offline):
            raise SystemExit(
                entry["id"]
                + ": choose --allow-network for the first download or --offline for cached assets"
            )
        project = ROOT / manifest["environments"][entry["environment"]]["project"]
        python = project / ".venv/bin/python"
        if not python.exists():
            raise SystemExit(
                f"Missing environment. Run uv sync --locked --project {project.relative_to(ROOT)}"
            )
        actual = subprocess.check_output(
            [
                str(python),
                "-c",
                'import sys;print(".".join(map(str,sys.version_info[:3])))',
            ],
            text=True,
        ).strip()
        if not actual.startswith("3.11."):
            raise SystemExit("Expected Python 3.11; found " + actual)
        frozen = json.loads(
            subprocess.check_output(
                [
                    str(python),
                    "-c",
                    'import importlib.metadata as m,json; print(json.dumps({d.metadata["Name"]:d.version for d in m.distributions()}))',
                ],
                text=True,
            )
        )
        lock_probe = 'import json,tomllib,sys;d=tomllib.load(open(sys.argv[1],"rb"));p=tomllib.load(open(sys.argv[2],"rb"));print(json.dumps({"versions":{x["name"]:[y["version"] for y in d["package"] if y["name"]==x["name"]] for x in d["package"]},"direct":p["project"]["dependencies"]}))'
        locked = json.loads(
            subprocess.check_output(
                [
                    str(python),
                    "-c",
                    lock_probe,
                    str(project / "uv.lock"),
                    str(project / "pyproject.toml"),
                ],
                text=True,
            )
        )
        normalize = lambda name: name.lower().replace("_", "-")
        installed = {normalize(name): version for name, version in frozen.items()}
        for name, version in installed.items():
            if version not in locked["versions"].get(name, []):
                raise SystemExit(f"Environment drift: {name}; run uv sync --locked")
        for requirement in locked["direct"]:
            name, version = requirement.split("==")
            if installed.get(normalize(name), "").split("+")[0] != version:
                raise SystemExit(f"Missing dependency: {name}; run uv sync --locked")
        command = [str(python), entry["entrypoint"]]
        env = os.environ.copy()
        env.update(
            {
                "PYTHONHASHSEED": "0",
                "TOKENIZERS_PARALLELISM": "false",
                "HF_HUB_DISABLE_PROGRESS_BARS": "1",
            }
        )
        if not args.allow_network:
            env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
        start = time.monotonic()
        status = "passed"
        error = None
        stdout = ""
        stderr = ""
        try:
            run = subprocess.run(
                command,
                cwd=ROOT / entry["workingDirectory"],
                env=env,
                capture_output=True,
                text=True,
                timeout=entry["runtime"]["timeoutSeconds"],
            )
            stdout, stderr = (
                run.stdout.replace(str(ROOT), "<repository>"),
                run.stderr.replace(str(ROOT), "<repository>"),
            )
            if run.returncode:
                raise ValueError(f"process exit {run.returncode}")
            check = entry["validation"]
            if (
                check["method"] == "exact-stdout"
                and stdout.strip() != check["expected"].strip()
            ):
                raise ValueError("stdout differs from expected output")
            if check["method"] == "json-invariants":
                result = json.loads(stdout)
                for key in ["shape", "top_documents"]:
                    if result[key] != check[key]:
                        raise ValueError(f"{key} invariant failed")
        except (ValueError, subprocess.TimeoutExpired) as exc:
            status = "failed"
            error = str(exc)
            failures.append(entry["id"])
        elapsed = round(time.monotonic() - start, 3)
        # RUSAGE_CHILDREN is a cumulative maximum across this runner's children;
        # records therefore conservatively include interpreter/package probes.
        usage = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        memory = round(usage / (1024 * 1024 if sys.platform == "darwin" else 1024), 1)
        record = {
            "schemaVersion": 1,
            "exampleId": entry["id"],
            "status": status,
            "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "gitCommit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "workingTreeDirty": bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain"], cwd=ROOT, text=True
                ).strip()
            ),
            "command": ["<environment-python>", entry["entrypoint"]],
            "sourceSha256": digest(ROOT / entry["entrypoint"]),
            "lockSha256": digest(project / "uv.lock"),
            "contractSha256": contract_digest(entry),
            "fixtureSha256": json.loads((ROOT / entry["dataset"]).read_text())["sha256"]
            if entry["dataset"]
            else None,
            "models": entry["models"],
            "python": actual,
            "packages": frozen,
            "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
            "device": "cpu",
            "durationSeconds": elapsed,
            "peakMemoryMiB": memory,
            "memoryMeasurement": "RUSAGE_CHILDREN cumulative peak; includes environment probes",
            "networkMode": "allowed" if args.allow_network else "offline-cache",
            "stdout": stdout,
            "stderr": stderr,
            "error": error,
        }
        if args.record:
            directory = ROOT / "examples/verification/records"
            directory.mkdir(parents=True, exist_ok=True)
            name = entry["id"] + ("" if status == "passed" else "-failed") + ".json"
            path = directory / name
            path.write_text(json.dumps(record, indent=2) + "\n")
            if status == "passed":
                entry["verificationRecord"] = str(path.relative_to(ROOT))
        print(f"{entry['id']}: {status} ({elapsed}s)")
        if error:
            print(error, stderr[-1500:], file=sys.stderr)
    if args.record:
        (ROOT / "examples/manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
    print(f"{len(selected) - len(failures)}/{len(selected)} examples passed")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
