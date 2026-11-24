"""Offline grading template/import utility. This script performs no network requests."""

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "examples/foundations/src"))
from atlas_foundations.e8.contracts import output_hash, validate_judgments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=ROOT / "examples/artifacts/e8-evaluation/results.json",
    )
    parser.add_argument(
        "--template",
        type=Path,
        help="Write a blank randomized grading packet with blinded model codes.",
    )
    parser.add_argument(
        "--ratings",
        type=Path,
        help="Read completed normalized rows; never overwrite the public snapshot.",
    )
    parser.add_argument(
        "--compare-order",
        type=Path,
        help="Optional second independently graded AB/BA packet to report order disagreements.",
    )
    parser.add_argument('--unblind-map', type=Path, help='Join a completed blinded packet to its private system/id mapping.')
    args = parser.parse_args()
    result = json.loads(args.results.read_text())
    rows = [dict(r, model=run["model"]) for run in result["runs"] for r in run["rows"]]
    if args.template:
        rng = random.Random(42)
        codes = {
            run["model"]: f"system-{i + 1}" for i, run in enumerate(result["runs"])
        }
        rng.shuffle(rows)
        packet = []
        for i, r in enumerate(rows):
            order = "AB" if i % 2 == 0 else "BA"
            packet.append(
                dict(
                    system=codes[r["model"]],
                    id=r["id"],
                    context=r["context"],
                    task_instruction=r["instruction"],
                    answer_a=r["prediction"] if order=='AB' else r["reference"],
                    answer_b=r["reference"] if order=='AB' else r["prediction"],
                    reference=r["reference"],
                    prediction=r["prediction"],
                    model=r["model"],
                    output_sha256=output_hash(r["prediction"]),
                    rubric_version="atlas-e8-v1",
                    judge_version="",
                    judge_prompt_version="",
                    judge_prompt_sha256="",
                    order=order,
                    grounding=None,
                    instruction=None,
                    clarity=None,
                )
            )
        # Strip model identities from the rater copy; retain this private mapping separately.
        mapping = [
            dict(system=r["system"], id=r["id"], model=r.pop("model")) for r in packet
        ]
        args.template.write_text(
            json.dumps(packet, indent=2, ensure_ascii=False) + "\n"
        )
        args.template.with_suffix(".mapping.json").write_text(
            json.dumps(mapping, indent=2) + "\n"
        )
        print(
            "Wrote blank grading packet and separate unblinding map. Keep mapping hidden from raters; no ratings were generated."
        )
    def read_ratings(path):
        values=json.loads(path.read_text())
        if args.unblind_map:
            mapping={(r['system'],r['id']):r['model'] for r in json.loads(args.unblind_map.read_text())}
            for row in values:
                row['model']=mapping[(row['system'],row['id'])]
        return values

    if args.ratings:
        ratings = read_ratings(args.ratings)
        print(json.dumps(validate_judgments(rows, ratings), indent=2))
        if args.compare_order:
            other = read_ratings(args.compare_order)
            validate_judgments(rows, other)
            by = {(r["model"], r["id"]): r for r in other}
            matched = 0
            disagreements = 0
            for r in ratings:
                peer = by.get((r["model"], r["id"]))
                if peer is not None:
                    if peer["order"] == r["order"]:
                        raise ValueError(
                            "Order comparison requires opposite presentation order"
                        )
                    matched += 1
                    disagreements += any(
                        r[k] != peer[k] for k in ["grounding", "instruction", "clarity"]
                    )
            print(
                json.dumps(
                    dict(order_matched=matched, order_score_disagreements=disagreements)
                )
            )
    if not (args.ratings or args.template):
        parser.error("Choose --template or --ratings")


if __name__ == "__main__":
    main()
