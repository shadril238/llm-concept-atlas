"""Train a tiny BERT token classifier with explicit word/subword contracts."""

from collections import Counter
import json
from pathlib import Path
import sys
import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
)

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "examples/foundations/src"))
from atlas_foundations.ner_contract import LABELS, align_first, entity_scores, spans

MODEL = "prajjwal1/bert-tiny"
REVISION = "6f75de8b60a9f8a2fdf7b69cbd86d9e64bcb3837"
TOKENIZER = "google-bert/bert-base-uncased"
TOKENIZER_REVISION = "86b5e0934494bd15c9632b12f734a8a67f723594"


def main(output_path=None, model_path=None, tokenizer_path=None):
    torch.manual_seed(7)
    torch.set_num_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path or TOKENIZER,
        revision=None if tokenizer_path else TOKENIZER_REVISION,
        use_fast=True,
    )
    model = AutoModelForTokenClassification.from_pretrained(
        model_path or MODEL,
        revision=None if model_path else REVISION,
        num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)),
        label2id={tag: i for i, tag in enumerate(LABELS)},
    )
    assert model.config.vocab_size == tokenizer.vocab_size == 30522
    records = json.loads(
        (ROOT / "examples/data/fixtures/ner-reference.json").read_text()
    )["records"]
    splits = {
        key: [r for r in records if r["split"] == key]
        for key in ["train", "validation", "test"]
    }
    collate = DataCollatorForTokenClassification(
        tokenizer, padding=True, label_pad_token_id=-100
    )

    # region alignment
    def encode(row):
        complete = tokenizer(row["tokens"], is_split_into_words=True, truncation=False)
        counts = Counter(i for i in complete.word_ids() if i is not None)
        encoded = tokenizer(
            row["tokens"], is_split_into_words=True, truncation=True, max_length=48
        )
        encoded["labels"] = align_first(encoded.word_ids(), row["tags"], counts)
        assert len(encoded["labels"]) == len(encoded["input_ids"])
        return dict(encoded)

    # endregion alignment
    prepared = {key: [encode(row) for row in rows] for key, rows in splits.items()}

    def predict(net, rows):
        net.eval()
        predicted = []
        with torch.no_grad():
            for row in rows:
                encoded = tokenizer(
                    row["tokens"], is_split_into_words=True, return_tensors="pt"
                )
                indices = net(**encoded).logits.argmax(-1)[0].tolist()
                seen = set()
                tags = []
                for index, word in zip(indices, encoded.word_ids()):
                    if word is not None and word not in seen:
                        tags.append(LABELS[index])
                        seen.add(word)
                assert len(tags) == len(row["tokens"])
                predicted.append(tags)
        return predicted

    # Freeze embeddings only; both encoder layers and the new head train.
    for parameter in model.bert.embeddings.parameters():
        parameter.requires_grad = False
    frozen = {
        name: p.detach().clone()
        for name, p in model.named_parameters()
        if not p.requires_grad
    }
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=0.001
    )
    path = Path(output_path) if output_path else ROOT / "examples/artifacts/ner-model"
    path.mkdir(parents=True, exist_ok=True)
    best = -1.0
    history = []
    for step in range(90):
        model.train()
        optimizer.zero_grad()
        batch = collate(
            [
                prepared["train"][(step * 8 + i) % len(prepared["train"])]
                for i in range(8)
            ]
        )
        loss = model(**batch).loss
        assert torch.isfinite(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if (step + 1) % 30 == 0:
            predicted = predict(model, splits["validation"])
            metric = entity_scores([r["tags"] for r in splits["validation"]], predicted)
            history.append(
                {
                    "step": step + 1,
                    "train_loss": loss.item(),
                    "validation_f1": metric["f1"],
                }
            )
            if metric["f1"] > best:
                best = metric["f1"]
                model.save_pretrained(path / "model")
                tokenizer.save_pretrained(path / "model")
    assert all(
        torch.equal(frozen[name], p)
        for name, p in model.named_parameters()
        if name in frozen
    )
    reloaded = AutoModelForTokenClassification.from_pretrained(path / "model").eval()
    second = AutoModelForTokenClassification.from_pretrained(path / "model").eval()
    predicted = predict(reloaded, splits["test"])
    assert predicted == predict(second, splits["test"])
    validation_predictions = predict(reloaded, splits["validation"])
    validation_errors = [
        {"tokens": r["tokens"], "gold": r["tags"], "predicted": p}
        for r, p in zip(splits["validation"], validation_predictions)
        if r["tags"] != p
    ]
    gold = [row["tags"] for row in splits["test"]]
    baseline = entity_scores(gold, [["O"] * len(tags) for tags in gold])
    metrics = entity_scores(gold, predicted)
    accuracy = sum(
        a == b for aa, bb in zip(gold, predicted) for a, b in zip(aa, bb)
    ) / sum(map(len, gold))
    errors = [
        {
            "id": r["id"],
            "tokens": r["tokens"],
            "gold": r["tags"],
            "predicted": p,
            "gold_spans": spans(r["tags"]),
            "predicted_spans": spans(p),
        }
        for r, p in zip(splits["test"], predicted)
    ]
    row = splits["train"][0]
    encoded = tokenizer(
        row["tokens"], is_split_into_words=True, padding="max_length", max_length=24
    )
    counts = Counter(i for i in encoded.word_ids() if i is not None)
    labels = align_first(encoded.word_ids(), row["tags"], counts)
    alignment = [
        {
            "wordId": w,
            "word": row["tokens"][w] if w is not None else "",
            "wordTag": row["tags"][w] if w is not None else "",
            "piece": piece,
            "id": index,
            "label": label,
        }
        for w, piece, index, label in zip(
            encoded.word_ids(),
            tokenizer.convert_ids_to_tokens(encoded.input_ids),
            encoded.input_ids,
            labels,
        )
    ]
    assert any(count > 1 for count in counts.values()), (
        "Fixture must expose split words"
    )
    result = {
        "metrics": metrics,
        "all_o_baseline": baseline,
        "token_accuracy": accuracy,
        "history": history,
        "predictions": errors,
        "validation_errors": validation_errors,
        "alignment": alignment,
        "frozen_embeddings": True,
        "reload": True,
        "train_documents": len(splits["train"]),
        "validation_documents": len(splits["validation"]),
        "test_documents": len(splits["test"]),
    }
    (path / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in ["alignment", "predictions", "validation_errors"]
            }
        )
    )
    return result


if __name__ == "__main__":
    main()
