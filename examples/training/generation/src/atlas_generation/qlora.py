"""L21 optional genuine Linux CPU NF4/double-quant adapter run; CUDA profile unverified.

This is a small algorithm verification, NOT the book's large-model paged CUDA profile.
The separate quantized environment contains bitsandbytes; the Mac environment does not.
"""

import json
import copy
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(
    0, str(Path(__file__).resolve().parents[5] / "examples/foundations/src")
)
from peft import PeftModel

from atlas_generation.common import (
    MODEL,
    ROOT,
    AutoModelForCausalLM,
    adapter,
    artifacts,
    base,
    batch,
    data,
    download,
    fit_sft,
    generate,
    nll,
    save,
    seed,
    tokenizer,
    torch,
)
from peft import prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig


def quantized():
    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float32,
    )
    return AutoModelForCausalLM.from_pretrained(
        download(MODEL),
        local_files_only=True,
        torch_dtype=torch.float32,
        quantization_config=config,
        device_map={"": "cpu"},
    )


def main():
    if platform.system() != "Linux":
        raise RuntimeError(
            "Verified target is Linux CPU; macOS/MPS and CUDA are not certified by this entrypoint."
        )
    import bitsandbytes as bnb

    seed()
    tok = tokenizer()
    rows = data()
    test = [r for r in rows if r["split"] == "test"]
    q = quantized()
    layers = [(n, m) for n, m in q.named_modules() if isinstance(m, bnb.nn.Linear4bit)]
    assert layers and all(
        m.weight.quant_state.quant_type == "nf4" and m.weight.quant_state.nested
        for _, m in layers
    )
    q = prepare_model_for_kbit_training(q, use_gradient_checkpointing=False)
    model = adapter(q)
    baseline = dict(nll=nll(model, tok, test), generations=generate(model, tok, test))
    report = fit_sft(model, tok, rows, 20, 0.003)
    assert all("lora_" in n for n, p in model.named_parameters() if p.requires_grad)
    folder = ROOT / "examples/artifacts/e8-qlora"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "adapter"
    model.save_pretrained(path)
    tok.save_pretrained(path)
    b = batch(tok, test[:1])
    model.eval()
    with torch.no_grad():
        expected = model(**b).logits
    fresh = PeftModel.from_pretrained(
        prepare_model_for_kbit_training(quantized(), use_gradient_checkpointing=False),
        path,
        torch_device="cpu",
    ).eval()
    with torch.no_grad():
        reload_error = float((expected - fresh(**b).logits).abs().max())
    assert reload_error < 1e-5
    # Dequantize a fresh base FIRST, then wrap/load adapter and merge. This preserves
    # the quantized approximation; it does not recover original pretrained weights.
    floating = quantized()
    floating.dequantize()
    floating = PeftModel.from_pretrained(floating, path, torch_device="cpu").eval()
    with torch.no_grad():
        dequant_error = float((expected - floating(**b).logits).abs().max())
        dequant_logits = floating(**b).logits
    algebra = copy.deepcopy(floating).double().eval()
    with torch.no_grad():
        algebra_expected = algebra(**b).logits.clone()
    algebra = algebra.merge_and_unload().eval()
    with torch.no_grad():
        float64_error = float((algebra_expected - algebra(**b).logits).abs().max())
    assert float64_error < 1e-8
    merged = floating.merge_and_unload().eval()
    with torch.no_grad():
        merge_error = float((dequant_logits - merged(**b).logits).abs().max())
    scale = float(expected.abs().max())
    tolerance = 0.001 * max(1.0, scale)
    # Floating summation through this tiny normalized model can amplify rounding.
    # Bound observed runtime drift separately from the independent double algebra.
    print(
        json.dumps(
            dict(
                dequant_error=dequant_error,
                merge_error=merge_error,
                scale=scale,
                tolerance=tolerance,
                float64_error=float64_error,
            )
        ),
        file=sys.stderr,
    )
    assert dequant_error < tolerance and merge_error < tolerance
    merged_path = folder / "merged-dequantized"
    merged.save_pretrained(merged_path)
    tok.save_pretrained(merged_path)
    restored = base(merged_path).eval()
    with torch.no_grad():
        merged_reload_error = float(
            (merged(**b).logits - restored(**b).logits).abs().max()
        )
    assert merged_reload_error < 1e-5
    result = dict(
        backend=platform.platform(),
        quantizer="bitsandbytes0.48.2 NF4 + nested/double quantization",
        compute_dtype="float32",
        adapter_dtype="float32",
        optimizer="torch AdamW; NOT paged optimizer",
        gpu_profile="unverified; no CUDA run",
        linear4bit_layers=[n for n, _ in layers],
        baseline=baseline,
        training=report,
        test_nll=nll(model, tok, test),
        generations=generate(model, tok, test),
        reload_max_abs_error=reload_error,
        dequant_max_abs_error=dequant_error,
        merge_max_abs_error=merge_error,
        float32_global_logit_scale=scale,
        float32_relative_tolerance=0.001,
        float64_merge_max_abs_error=float64_error,
        merged_reload_max_abs_error=merged_reload_error,
        merge_scope="Dequantized NF4 base + adapter; original high-precision base is a different model. No requantized merge claim.",
        artifacts=artifacts(folder),
    )
    save("e8-qlora", result)
    print(
        json.dumps(
            dict(
                nf4_verified=True,
                double_quant_verified=True,
                adapters_trained=True,
                reload_verified=True,
                merge_verified=True,
            )
        )
    )
    return result


if __name__ == "__main__":
    main()
