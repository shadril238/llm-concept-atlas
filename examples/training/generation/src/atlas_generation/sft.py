"""L21: actual full/LoRA completion-only instruction tuning and artifact checks."""

import gc
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(
    0, str(Path(__file__).resolve().parents[5] / "examples/foundations/src")
)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from download_models import SPECS

from peft import PeftModel

from atlas_generation.common import (
    MODEL,
    ROOT,
    adapter,
    artifacts,
    base,
    batch,
    data,
    encode,
    fit_sft,
    generate,
    nll,
    save,
    seed,
    tokenizer,
    torch,
)


def main():
    seed()
    rows = data()
    tok = tokenizer()
    test = [r for r in rows if r["split"] == "test"]
    for r in rows:
        encode(tok, r)
    folder = ROOT / "examples/artifacts/e8-sft"
    folder.mkdir(parents=True, exist_ok=True)
    model = base()
    baseline = dict(nll=nll(model, tok, test), generations=generate(model, tok, test))
    del model
    gc.collect()
    runs = []
    for kind, steps, lr in [("full", 40, 0.0002), ("lora", 60, 0.003)]:
        seed()
        model = base()
        if kind == "lora":
            model = adapter(model)
        report = fit_sft(model, tok, rows, steps, lr)
        report.update(
            kind=kind,
            test_nll=nll(model, tok, test),
            generations=generate(model, tok, test),
        )
        path = folder / kind
        model.save_pretrained(path)
        tok.save_pretrained(path)
        probe = batch(tok, test[:1])
        model.eval()
        with torch.no_grad():
            expected = model(**probe).logits
        reloaded = (
            base(path)
            if kind == "full"
            else PeftModel.from_pretrained(base(), path).eval()
        )
        with torch.no_grad():
            actual = reloaded(**probe).logits
        delta = float((expected - actual).abs().max())
        assert delta < 1e-5
        report["reload_max_abs_error"] = delta
        if kind == "lora":
            merged = reloaded.merge_and_unload().eval()
            with torch.no_grad():
                difference = float((expected - merged(**probe).logits).abs().max())
                scale = float(expected.abs().max())
            # Pythia's small normalization layers amplify float32 summation differences.
            # Verify exact algebra separately in float64 and bound runtime drift.
            algebra = PeftModel.from_pretrained(base().double(), path).double().eval()
            with torch.no_grad():
                double_expected = algebra(**probe).logits.clone()
            algebra = algebra.merge_and_unload().eval()
            with torch.no_grad():
                double_error = float(
                    (double_expected - algebra(**probe).logits).abs().max()
                )
            assert double_error < 1e-8
            assert difference < 1e-3 * max(1.0, scale)
            del algebra
            merged_path = folder / "merged"
            merged.save_pretrained(merged_path)
            tok.save_pretrained(merged_path)
            identity = dict(
                base=MODEL,
                revision=SPECS[MODEL]["revision"],
                adapter="lora",
                dtype="float32",
                template="atlas-completion-v1",
                rank=4,
                alpha=8,
                target="query_key_value",
                selected_step=report["selected_step"],
            )
            (merged_path / "atlas-identity.json").write_text(
                json.dumps(identity, indent=2) + "\n"
            )
            # A saved merged model must also survive a fresh load, not only in-memory merging.
            fresh = base(merged_path).eval()
            with torch.no_grad():
                reload_delta = float(
                    (merged(**probe).logits - fresh(**probe).logits).abs().max()
                )
            assert reload_delta < 1e-5
            report.update(
                merge_max_abs_error=difference,
                merge_global_scale=scale,
                float32_relative_tolerance=0.001,
                float64_merge_max_abs_error=double_error,
                merged_reload_max_abs_error=reload_delta,
            )
            del merged, fresh
        runs.append(report)
        del model, reloaded
        gc.collect()
    inspected = []
    for r in json.loads(
        (ROOT / "examples/data/fixtures/instruction-contract.json").read_text()
    )["rows"]:
        b = encode(tok, r)
        inspected.append(
            dict(id=r["id"], tokens=tok.convert_ids_to_tokens(b["input_ids"]), **b)
        )
    result = dict(
        model=MODEL,
        revision=SPECS[MODEL]["revision"],
        backend="torch float32 CPU; one thread",
        template="Original ordinary-text completion delimiters; Pythia has no native chat template.",
        baseline=baseline,
        runs=runs,
        inspection=inspected,
        artifacts=artifacts(folder),
        gpu_profile="unverified; no supported CUDA run available",
        limits="48 filtered extraction rows, tiny pretrained model, 24 generated tokens; not a benchmark or proof of instruction-following quality.",
    )
    save("e8-sft", result)
    print(
        json.dumps(
            dict(
                training_updates_verified=all(r["changed"] for r in runs),
                reload_verified=True,
                merge_verified=True,
                masks_verified=all(
                    x["labels"][-1] == tok.eos_token_id for x in inspected
                ),
            )
        )
    )
    return result


if __name__ == "__main__":
    main()
