"""L19: genuine dynamic MLM adaptation, fixed masked validation and downstream comparison."""

from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atlas_representation.common import (
    AutoTokenizer,
    MODEL,
    ROOT,
    artifact_files,
    classifier_run,
    copy,
    download,
    random,
    save,
    seed,
    sentiment,
    state_hash,
    torch,
)
from transformers import AutoModelForMaskedLM, DataCollatorForLanguageModeling


def main():
    source = sentiment()["rows"]
    train = [r for r in source if r["split"] == "train"]
    valid = [r for r in source if r["split"] == "validation"]
    test = [r for r in source if r["split"] == "test"]
    tokenizer = AutoTokenizer.from_pretrained(download(MODEL), local_files_only=True)
    encode = lambda rows: [
        tokenizer(
            r["text"], truncation=True, max_length=96, return_special_tokens_mask=True
        )
        for r in rows
    ]
    training = encode(train)
    validation = encode(valid)
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15, seed=999
    )
    fixed = [collator(validation[i : i + 8]) for i in range(0, len(validation), 8)]
    assert all((b["labels"] != -100).any() for b in fixed)

    def evaluate(model):
        model.eval()
        total = 0.0
        tokens = 0
        with torch.no_grad():
            for b in fixed:
                count = int((b["labels"] != -100).sum())
                total += float(model(**b).loss) * count
                tokens += count
        return total / tokens

    def adapt(steps, name, keep=False):
        seed()
        model = AutoModelForMaskedLM.from_pretrained(
            download(MODEL), local_files_only=True
        ).cpu()
        initial = state_hash(model)
        baseline = evaluate(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        dynamic = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, mlm_probability=0.15, seed=42
        )
        rng = random.Random(42)
        history = []
        best = None
        score = None
        grad = None
        for step in range(steps):
            batch = dynamic(rng.sample(training, 8))
            assert (batch["labels"] != -100).any()
            model.train()
            optimizer.zero_grad()
            loss = model(**batch).loss
            assert torch.isfinite(loss)
            loss.backward()
            if step == 0:
                grad = float(
                    model.bert.encoder.layer[0].attention.self.query.weight.grad.norm()
                )
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            if (step + 1) % 10 == 0 or step == steps - 1:
                value = evaluate(model)
                history.append(
                    dict(
                        step=step + 1,
                        training_loss=float(loss.detach()),
                        validation_masked_loss=value,
                    )
                )
                if score is None or value < score:
                    score = value
                    best = copy.deepcopy(model.state_dict())
                    best_step = step + 1
        model.load_state_dict(best)
        assert state_hash(model) != initial and grad > 0
        directory = ROOT / f"examples/artifacts/e7-mlm/{name}"
        model.save_pretrained(directory)
        tokenizer.save_pretrained(directory)
        reloaded = AutoModelForMaskedLM.from_pretrained(directory).eval()
        assert abs(evaluate(model) - evaluate(reloaded)) < 1e-6
        sentence = "The acting was [MASK]."
        probe = tokenizer(sentence, return_tensors="pt")
        position = int((probe["input_ids"][0] == tokenizer.mask_token_id).nonzero()[0])
        with torch.no_grad():
            probs = reloaded(**probe).logits[0, position].softmax(-1)
            values, ids = probs.topk(5)
        return dict(
            profile=name,
            steps=steps,
            baseline_masked_validation_loss=baseline,
            selected_validation_loss=score,
            selected_step=best_step,
            history=history,
            initial_sha256=initial,
            updated_sha256=state_hash(model),
            first_query_gradient_norm=grad,
            reload_verified=True,
            fill_mask=dict(
                input=sentence,
                top=[
                    dict(
                        token=tokenizer.convert_ids_to_tokens(int(i)),
                        probability=float(v),
                    )
                    for i, v in zip(ids, values)
                ],
            ),
            artifact=artifact_files(directory),
        ), directory

    smoke, _ = adapt(2, "smoke-2-steps")
    evaluated, path = adapt(60, "evaluated-60-steps")
    # Equal downstream data, step budget and random head seed; the adapted encoder is the intended intervention.
    downstream = []
    for name, backbone in [("pretrained", None), ("mlm-adapted", str(path))]:
        directory = ROOT / f"examples/artifacts/e7-mlm/downstream-{name}"
        row = classifier_run(
            train, valid, test, directory, backbone=backbone, run_seed=42, steps=60
        )
        row["condition"] = name
        downstream.append(row)
    assert (
        downstream[0]["head_pooler_initial_sha256"]
        == downstream[1]["head_pooler_initial_sha256"]
    )
    original = tokenizer.pad(validation[:8], padding=True, return_tensors="pt")
    b = fixed[0]
    tokens = []
    for i, (orig, changed, label, mask) in enumerate(
        zip(
            original["input_ids"][0],
            b["input_ids"][0],
            b["labels"][0],
            original["special_tokens_mask"][0],
        )
    ):
        if int(label) != -100:
            assert not int(mask) and int(label) == int(orig)
        tokens.append(
            dict(
                position=i,
                original_id=int(orig),
                input_id=int(changed),
                label=int(label),
                original_token=tokenizer.convert_ids_to_tokens(int(orig)),
                input_token=tokenizer.convert_ids_to_tokens(int(changed)),
                special_or_padding=bool(mask) or int(orig) == tokenizer.pad_token_id,
                loss_included=int(label) != -100,
            )
        )
    save(
        "e7-mlm",
        dict(
            smoke=smoke,
            evaluated=evaluated,
            downstream=downstream,
            mask_trace=tokens,
            training_ids=[r["id"] for r in train],
            validation_ids=[r["id"] for r in valid],
            test_ids=[r["id"] for r in test],
            protocol="Only 200 training review texts enter continued pretraining, never labels/test/challenge texts. Fixed dynamically-created validation masks select minimum validation masked-token loss; 60-step evaluated run is separate from a two-step smoke run. Downstream runs have identical data/seed/budget and validation-only classification selection. Pretraining overlap of the original downloaded checkpoint is unknown.",
            masking="Actual Transformers collator: 15% eligible token selection, nominal 80% mask / 10% random vocabulary / 10% unchanged among selected tokens. Observed unchanged IDs do not identify which random branch produced them. Special/padding labels remain -100.",
            verified=True,
        ),
    )
    print(
        json.dumps(
            dict(
                mlm_updated=True,
                test_excluded=True,
                downstream_compared=True,
                reload_verified=True,
            )
        )
    )


if __name__ == "__main__":
    main()
