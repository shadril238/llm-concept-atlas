"""Shared transparent CPU loops. No trainer defaults hide the loss mask or reference."""

import hashlib
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
PROJECT = ROOT / "examples/training/generation"
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(ROOT / "examples/foundations/src"))
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import torch.nn.functional as F
from atlas_foundations.e8.contracts import check_data, check_preferences, encode, prompt
from download_models import download
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "EleutherAI/pythia-14m"
torch.set_num_threads(1)


def seed():
    random.seed(42)
    torch.manual_seed(42)


def data():
    rows = json.loads(
        (ROOT / "examples/data/fixtures/instruction-dolly.json").read_text()
    )["rows"]
    check_data(rows)
    return rows


def preferences():
    rows = json.loads(
        (ROOT / "examples/data/fixtures/generation-preferences.json").read_text()
    )["rows"]
    check_preferences(rows)
    return rows


def tokenizer():
    tok = AutoTokenizer.from_pretrained(download(MODEL), local_files_only=True)
    tok.pad_token = tok.eos_token
    return tok


def base(path=None):
    return AutoModelForCausalLM.from_pretrained(
        path or download(MODEL), local_files_only=True, torch_dtype=torch.float32
    ).cpu()


def adapter(model):
    return get_peft_model(
        model,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            target_modules=["query_key_value"],
            bias="none",
        ),
    )


def batch(tok, rows, key=None):
    items = [encode(tok, r, r[key] if key else None) for r in rows]
    width = max(len(x["input_ids"]) for x in items)
    return {
        k: torch.tensor(
            [
                x[k]
                + [
                    tok.pad_token_id
                    if k == "input_ids"
                    else -100
                    if k == "labels"
                    else 0
                ]
                * (width - len(x[k]))
                for x in items
            ]
        )
        for k in ["input_ids", "labels", "attention_mask"]
    }


def logps(model, b):
    logits = (
        model(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
        .logits[:, :-1]
        .float()
    )
    labels = b["labels"][:, 1:]
    mask = labels.ne(-100)
    gathered = (
        F.log_softmax(logits, -1)
        .gather(-1, labels.clamp_min(0).unsqueeze(-1))
        .squeeze(-1)
    )
    return (gathered * mask).sum(-1), mask.sum(-1)


def nll(model, tok, rows):
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for r in rows:
            p, n = logps(model, batch(tok, [r]))
            total -= p.item()
            count += n.item()
    return total / count


def digest(model, frozen=False):
    h = hashlib.sha256()
    for n, p in model.named_parameters():
        if not frozen or not p.requires_grad:
            h.update(n.encode())
            h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def generate(model, tok, rows):
    model.eval()
    out = []
    with torch.no_grad():
        for r in rows:
            ids = tok.encode(prompt(r), add_special_tokens=False, return_tensors="pt")
            seq = model.generate(
                ids,
                attention_mask=torch.ones_like(ids),
                max_new_tokens=24,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
            completion = seq[0, ids.shape[1] :]
            prediction = tok.decode(completion, skip_special_tokens=True)
            out.append(
                dict(
                    id=r["id"],
                    prediction=prediction,
                    reference=r["response"],
                    context=r["context"],
                    instruction=r["instruction"],
                    generated_tokens=len(completion),
                    hit_limit=bool(
                        len(completion) == 24 and completion[-1] != tok.eos_token_id
                    ),
                )
            )
    return out


def save(name, result):
    p = ROOT / "examples/artifacts" / name
    p.mkdir(parents=True, exist_ok=True)
    (p / "results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )


def artifacts(p):
    return [
        dict(
            path=str(f.relative_to(ROOT)),
            bytes=f.stat().st_size,
            sha256=hashlib.sha256(f.read_bytes()).hexdigest(),
        )
        for f in sorted(p.rglob("*"))
        if f.is_file() and f.name != "results.json"
    ]


def logical_parameters(parameter):
    # Packed Params4bit.numel() counts storage elements, not logical model weights.
    state = getattr(parameter, "quant_state", None)
    if state is not None:
        import math

        return math.prod(state.shape)
    return parameter.numel()


def fit_sft(model, tok, rows, steps, lr):
    train = [r for r in rows if r["split"] == "train"]
    valid = [r for r in rows if r["split"] == "validation"]
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0
    )
    best = float("inf")
    state = {
        n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad
    }
    selected = 0
    trace = []
    initial = digest(model)
    frozen = digest(model, True)
    gradient = 0.0
    # Fixed cyclic batches, two microbatches per optimizer update (effective batch four).
    for step in range(1, steps + 1):
        model.train()
        optimizer.zero_grad()
        for micro in range(2):
            index = ((step - 1) * 4 + micro * 2) % len(train)
            b = batch(tok, [train[(index + j) % len(train)] for j in range(2)])
            loss = model(**b).loss / 2
            loss.backward()
        gradient = max(
            gradient,
            float(
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
            ),
        )
        optimizer.step()
        if step % 10 == 0:
            value = nll(model, tok, valid)
            trace.append(dict(step=step, validation_nll=value))
            if value < best:
                best = value
                selected = step
                state = {
                    n: p.detach().clone()
                    for n, p in model.named_parameters()
                    if p.requires_grad
                }
    # Restore only optimized tensors. Packed NF4 state_dict contains serialization
    # metadata that ordinary load_state_dict cannot round-trip in-place.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in state:
                parameter.copy_(state[name])
    model.eval()
    assert gradient > 0 and digest(model) != initial and selected > 0
    if any(not p.requires_grad for p in model.parameters()):
        assert frozen == digest(model, True)
    return dict(
        trace=trace,
        selected_step=selected,
        validation_nll=best,
        max_gradient_norm=gradient,
        changed=True,
        frozen_unchanged=frozen == digest(model, True),
        trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
        total=sum(logical_parameters(p) for p in model.parameters()),
        stored_elements=sum(p.numel() for p in model.parameters()),
        steps=steps,
        learning_rate=lr,
        microbatch=2,
        accumulation=2,
        max_length=256,
        optimizer="torch AdamW, constant LR, no warmup, weight_decay=0; clip norm1",
        selection="minimum validation token-weighted completion NLL among trained checkpoints; baseline reported separately, test untouched",
    )
