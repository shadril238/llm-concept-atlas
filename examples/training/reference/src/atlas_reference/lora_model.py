"""Pretrained 14M decoder + PEFT adapter; small CPU training demonstration."""

import hashlib
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel, get_peft_model

ROOT = Path(__file__).resolve().parents[5]
MODEL = "EleutherAI/pythia-14m"
REVISION = "cf967c0a9a04383db6f7b1108d86b2962634b4ac"


def frozen_digest(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().numpy().tobytes())
    return digest.hexdigest()


def main():
    torch.manual_seed(7)
    torch.set_num_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION).float()
    base.config.use_cache = False
    data = json.loads(
        (ROOT / "examples/data/fixtures/lora-reference.json").read_text()
    )["records"]
    splits = {
        split: [r for r in data if r["split"] == split]
        for split in ["train", "validation", "test"]
    }

    def batch(rows):
        texts = [r["prompt"] + r["answer"] + tokenizer.eos_token for r in rows]
        encoded = tokenizer(
            texts, padding=True, return_tensors="pt", add_special_tokens=False
        )
        assert encoded.input_ids.shape[1] <= 96, (
            "Inspect/truncate at task boundaries first"
        )
        labels = encoded.input_ids.clone()
        for i, row in enumerate(rows):
            prefix = tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]
            assert encoded.input_ids[i, : len(prefix)].tolist() == prefix, (
                "Prompt boundary changed tokenization"
            )
            labels[i, : len(prefix)] = -100
        labels[encoded.attention_mask == 0] = -100
        assert (labels != -100).sum() > 0
        return dict(encoded, labels=labels)

    def loss(model, rows):
        model.eval()
        with torch.no_grad():
            return model(**batch(rows)).loss.item()

    before = loss(base, splits["test"])
    # region adapter
    config = LoraConfig(
        task_type="CAUSAL_LM",
        r=4,
        lora_alpha=8,
        target_modules=["query_key_value"],
        lora_dropout=0.0,
        bias="none",
    )
    model = get_peft_model(base, config)
    # Only the fused Q/K/V projections get adapters in this architecture.
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=0.003
    )
    # endregion adapter
    frozen_before = frozen_digest(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    path = ROOT / "examples/artifacts/lora-model"
    path.mkdir(parents=True, exist_ok=True)
    history = []
    best = float("inf")
    for step in range(40):
        model.train()
        optimizer.zero_grad()
        rows = [
            splits["train"][(step * 4 + i) % len(splits["train"])] for i in range(4)
        ]
        objective = model(**batch(rows)).loss
        assert torch.isfinite(objective)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if (step + 1) % 10 == 0:
            validation = loss(model, splits["validation"])
            history.append(
                {
                    "step": step + 1,
                    "train_loss": objective.item(),
                    "validation_loss": validation,
                }
            )
            if validation < best:
                best = validation
                model.save_pretrained(path / "adapter")
    assert frozen_digest(model) == frozen_before
    tokenizer.save_pretrained(path / "adapter")
    reloaded = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION).float(),
        path / "adapter",
    ).eval()
    inputs = batch(splits["test"])
    with torch.no_grad():
        logits = reloaded(**inputs).logits
    after = loss(reloaded, splits["test"])
    # Reopen the saved adapter a second time and compare its inference.
    second = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION).float(),
        path / "adapter",
    ).eval()
    with torch.no_grad():
        assert torch.allclose(logits, second(**inputs).logits, atol=1e-5)
    merged = reloaded.merge_and_unload().eval()
    with torch.no_grad():
        merged_logits = merged(**inputs).logits
        error = (logits - merged_logits).abs().max().item()
        # Layerwise float32 rounding can amplify near-zero-logit relative error.
        assert error / max(1.0, logits.abs().max().item()) < 2e-4
    second = second.double().eval()
    with torch.no_grad():
        double_logits = second(**inputs).logits
    second = second.merge_and_unload().eval()
    with torch.no_grad():
        double_error = (double_logits - second(**inputs).logits).abs().max().item()
        assert double_error < 1e-9
    prompt = tokenizer(splits["test"][0]["prompt"], return_tensors="pt")
    with torch.no_grad():
        generated = merged.generate(
            **prompt,
            max_new_tokens=8,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    result = {
        "base_test_loss": before,
        "adapter_test_loss": after,
        "trainable": trainable,
        "total": total,
        "base_frozen": True,
        "reload_merge": True,
        "merge_max_error": error,
        "double_merge_max_error": double_error,
        "train_documents": len(splits["train"]),
        "validation_documents": len(splits["validation"]),
        "test_documents": len(splits["test"]),
        "history": history,
        "test_prompt": splits["test"][0]["prompt"],
        "generated_completion": tokenizer.decode(
            generated[0, prompt.input_ids.shape[1] :], skip_special_tokens=True
        ),
    }
    (path / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
