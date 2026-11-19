"""L06: cached/uncached correctness before a small warmed CPU timing."""

import json
from pathlib import Path
from time import perf_counter
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[6]
MODEL = "EleutherAI/pythia-14m"
REVISION = "cf967c0a9a04383db6f7b1108d86b2962634b4ac"


def main():
    torch.set_num_threads(1)
    torch.manual_seed(11)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL, revision=REVISION, attn_implementation="eager"
        )
        .double()
        .eval()
    )
    prompt = tokenizer("The sun rises in the", return_tensors="pt").input_ids
    forced = tokenizer(
        " east and the sky becomes bright.", add_special_tokens=False
    ).input_ids

    def run(cached):
        prefix = prompt.clone()
        values = []
        past = None
        out = None
        with torch.inference_mode():
            for step in range(len(forced) + 1):
                token_ids = prefix if not cached or step == 0 else prefix[:, -1:]
                out = model(
                    input_ids=token_ids,
                    past_key_values=past if cached else None,
                    attention_mask=torch.ones_like(prefix),
                    use_cache=cached,
                )
                values.append(out.logits[:, -1].clone())
                if cached:
                    past = out.past_key_values
                if step < len(forced):
                    prefix = torch.cat([prefix, torch.tensor([[forced[step]]])], dim=1)
        return values, past, prefix.shape[1]

    full, _, length = run(False)
    cached, past, _ = run(True)
    errors = [(a - b).abs().max().item() for a, b in zip(full, cached)]
    assert max(errors) < 1e-9, errors
    cache_layers = list(past)
    shapes = [list(key.shape) for key, value in cache_layers]
    assert all(shape[-2] == length for shape in shapes)
    measured_bytes = sum(
        t.numel() * t.element_size() for layer in cache_layers for t in layer
    )
    predicted_bytes = (
        2
        * model.config.num_hidden_layers
        * model.config.num_attention_heads
        * length
        * (model.config.hidden_size // model.config.num_attention_heads)
        * 8
    )
    assert measured_bytes == predicted_bytes
    timing = {}
    for mode in [False, True]:
        run(mode)  # warm-up, not included
        samples = []
        for _ in range(3):
            start = perf_counter()
            run(mode)
            samples.append(perf_counter() - start)
        timing["cached" if mode else "uncached"] = samples
    result = {
        "max_logit_error": max(errors),
        "cache_shape": shapes[0],
        "layers": len(shapes),
        "cache_bytes": measured_bytes,
        "positions": length,
        "forced_tokens": len(forced),
        "dtype": "float64",
        "cpu_seconds": timing,
        "correctness_checked": True,
        "memory_checked": True,
        "measurement": "CPU float64 correctness/timing demonstration, not a production-precision benchmark; warm-up then three same-token runs including prompt prefill. Tiny workloads can be slower with caching.",
    }
    path = ROOT / "examples/artifacts/e3-cache"
    path.mkdir(parents=True, exist_ok=True)
    (path / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
