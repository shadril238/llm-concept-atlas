"""L23: real scalar reward ranking, then separate policy/reference DPO optimization."""

import copy
import gc
import json
import math
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
    F,
    adapter,
    artifacts,
    base,
    batch,
    digest,
    download,
    generate,
    logps,
    nll,
    preferences,
    prompt,
    save,
    seed,
    tokenizer,
    torch,
)
from transformers import AutoModelForSequenceClassification


def reward_batch(tok, rows, key):
    texts = [prompt(r) + r[key] for r in rows]
    lengths = [len(tok.encode(t)) for t in texts]
    if max(lengths) > 256:
        raise ValueError("Reward input exceeds 256; silent truncation forbidden")
    return tok(texts, padding=True, return_tensors="pt")


def reward_eval(model, tok, rows):
    model.eval()
    out = []
    with torch.no_grad():
        for r in rows:
            c = model(**reward_batch(tok, [r], "chosen")).logits.item()
            d = model(**reward_batch(tok, [r], "rejected")).logits.item()
            out.append(
                dict(
                    id=r["id"],
                    instruction=r["instruction"],
                    context=r["context"],
                    chosen_text=r["chosen"],
                    rejected_text=r["rejected"],
                    chosen=c,
                    rejected=d,
                    correct=c > d,
                    tie=c == d,
                    error_type=r["error_type"],
                    chosen_tokens=len(tok.encode(r["chosen"])),
                    rejected_tokens=len(tok.encode(r["rejected"])),
                )
            )
    return dict(
        accuracy=sum(r["correct"] for r in out) / len(out),
        loss=sum(
            float(F.softplus(torch.tensor(r["rejected"] - r["chosen"]))) for r in out
        )
        / len(out),
        ties=sum(r["tie"] for r in out),
        rows=out,
        shorter_wins_baseline=sum(
            r["chosen_tokens"] < r["rejected_tokens"] for r in out
        )
        / len(out),
    )


def preference_eval(policy, reference, tok, rows):
    policy.eval()
    reference.eval()
    out = []
    with torch.no_grad():
        for r in rows:
            c = batch(tok, [r], "chosen")
            d = batch(tok, [r], "rejected")
            pc = logps(policy, c)[0].item()
            pd = logps(policy, d)[0].item()
            rc = logps(reference, c)[0].item()
            rd = logps(reference, d)[0].item()
            margin = (pc - pd) - (rc - rd)
            out.append(
                dict(
                    id=r["id"],
                    instruction=r["instruction"],
                    context=r["context"],
                    chosen_text=r["chosen"],
                    rejected_text=r["rejected"],
                    policy_chosen=pc,
                    policy_rejected=pd,
                    reference_chosen=rc,
                    reference_rejected=rd,
                    relative_margin=margin,
                    raw_prefers_chosen=pc > pd,
                    relative_correct=margin > 0,
                    tie=abs(margin) < 1e-6,
                    error_type=r["error_type"],
                )
            )
    return dict(
        raw_accuracy=sum(r["raw_prefers_chosen"] for r in out) / len(out),
        relative_accuracy=sum(r["relative_correct"] and not r["tie"] for r in out)
        / len(out),
        ties=sum(r["tie"] for r in out),
        loss=sum(
            float(F.softplus(torch.tensor(-0.1 * r["relative_margin"]))) for r in out
        )
        / len(out),
        rows=out,
    )


def main():
    seed()
    rows = preferences()
    train = [r for r in rows if r["split"] == "train"]
    valid = [r for r in rows if r["split"] == "validation"]
    test = [r for r in rows if r["split"] == "test"]
    folder = ROOT / "examples/artifacts/e8-preferences"
    folder.mkdir(parents=True, exist_ok=True)
    rt = AutoTokenizer.from_pretrained(
        download("prajjwal1/bert-tiny"), local_files_only=True
    )
    rm = AutoModelForSequenceClassification.from_pretrained(
        download("prajjwal1/bert-tiny"), num_labels=1, local_files_only=True
    ).cpu()
    before = digest(rm)
    baseline_rm = reward_eval(rm, rt, test)
    opt = torch.optim.AdamW(rm.parameters(), lr=0.0003, weight_decay=0.0)
    best = float("inf")
    state = copy.deepcopy(rm.state_dict())
    selected = 0
    trace = []
    for step in range(1, 61):
        rm.train()
        opt.zero_grad()
        sample = [train[((step - 1) * 4 + j) % len(train)] for j in range(4)]
        chosen = rm(**reward_batch(rt, sample, "chosen")).logits[:, 0]
        rejected = rm(**reward_batch(rt, sample, "rejected")).logits[:, 0]
        loss = -F.logsigmoid(chosen - rejected).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(rm.parameters(), 1.0)
        opt.step()
        if step % 10 == 0:
            v = reward_eval(rm, rt, valid)["loss"]
            trace.append(dict(step=step, validation_pair_loss=v))
            if v < best:
                best = v
                state = copy.deepcopy(rm.state_dict())
                selected = step
    rm.load_state_dict(state)
    rm.eval()
    assert digest(rm) != before and selected > 0
    reward = dict(
        baseline=baseline_rm,
        test=reward_eval(rm, rt, test),
        selected_step=selected,
        trace=trace,
        learning_rate=0.0003,
        steps=60,
        batch_size=4,
        model="prajjwal1/bert-tiny + new scalar head",
        selection="minimum validation pairwise logistic loss",
    )
    path = folder / "reward"
    rm.save_pretrained(path)
    rt.save_pretrained(path)
    fresh = AutoModelForSequenceClassification.from_pretrained(
        path, local_files_only=True
    )
    assert reward_eval(fresh, rt, test) == reward["test"]
    del rm, fresh, opt
    gc.collect()
    # The reward model is deliberately not consulted below: DPO uses sequence likelihoods.
    tok = tokenizer()
    sft = ROOT / "examples/artifacts/e8-sft/merged"
    identity = json.loads((sft / "atlas-identity.json").read_text())
    assert (
        identity["base"] == MODEL and identity["revision"] == SPECS[MODEL]["revision"]
    )
    reference = base(sft).eval()
    reference.requires_grad_(False)
    ref_hash = digest(reference)
    policy = adapter(base(sft))
    initial = preference_eval(policy, reference, tok, valid)
    assert (
        all(abs(r["relative_margin"]) < 1e-6 for r in initial["rows"])
        and abs(initial["loss"] - math.log(2)) < 1e-6
    )
    frozen = digest(policy, True)
    before = digest(policy)
    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad], lr=0.001, weight_decay=0.0
    )
    best = float("inf")
    state = copy.deepcopy(policy.state_dict())
    selected = 0
    trace = []
    for step in range(1, 41):
        policy.train()
        optimizer.zero_grad()
        sample = [train[((step - 1) * 2 + j) % len(train)] for j in range(2)]
        c = batch(tok, sample, "chosen")
        d = batch(tok, sample, "rejected")
        pc = logps(policy, c)[0]
        pd = logps(policy, d)[0]
        with torch.no_grad():
            rc = logps(reference, c)[0]
            rd = logps(reference, d)[0]
        loss = -F.logsigmoid(0.1 * ((pc - pd) - (rc - rd))).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in policy.parameters() if p.requires_grad], 1.0
        )
        optimizer.step()
        if step % 10 == 0:
            v = preference_eval(policy, reference, tok, valid)["loss"]
            trace.append(dict(step=step, validation_dpo_loss=v))
            if v < best:
                best = v
                state = copy.deepcopy(policy.state_dict())
                selected = step
    policy.load_state_dict(state)
    policy.eval()
    assert (
        digest(policy) != before
        and frozen == digest(policy, True)
        and digest(reference) == ref_hash
        and selected > 0
    )
    evaluated = preference_eval(policy, reference, tok, test)
    path = folder / "dpo"
    policy.save_pretrained(path)
    tok.save_pretrained(path)
    fresh = PeftModel.from_pretrained(base(sft), path, torch_device="cpu").eval()
    assert preference_eval(fresh, reference, tok, test) == evaluated
    generation = generate(policy, tok, test)
    original = base()
    base_pref = preference_eval(original, reference, tok, test)
    del original
    dpo = dict(
        test=evaluated,
        initial_validation=initial,
        sft_test=preference_eval(reference, reference, tok, test),
        base_test=base_pref,
        selected_step=selected,
        trace=trace,
        steps=40,
        learning_rate=0.001,
        batch_size=2,
        beta=0.1,
        loss="original sigmoid DPO; sum completion token logprobs including EOS; no length normalization",
        reference="frozen selected merged SFT LoRA checkpoint; separate policy initialized identically",
        test_nll=nll(policy, tok, test),
        generations=generation,
        reference_unchanged=True,
        frozen_unchanged=True,
        selection="minimum validation DPO loss; no test selection",
        trainable=sum(p.numel() for p in policy.parameters() if p.requires_grad),
    )
    result = dict(
        reward=reward,
        dpo=dpo,
        artifacts=artifacts(folder),
        limits="Rule-authored relative preferences, eight heldout prompts, tiny models. No PPO training, independent human labels, general alignment or safety claim.",
    )
    save("e8-preferences", result)
    print(
        json.dumps(
            dict(
                reward_trained=True,
                dpo_trained=True,
                reference_unchanged=True,
                reload_verified=True,
                initial_reference_verified=True,
            )
        )
    )
    return result


if __name__ == "__main__":
    main()
