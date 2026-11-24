"""L20: reuse the E2 trainer; extend annotation, alignment and strict entity-error evidence."""

from pathlib import Path
from collections import Counter
from contextlib import redirect_stdout
import io
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atlas_representation.common import (
    AutoTokenizer,
    MODEL,
    ROOT,
    TOKENIZER,
    artifact_files,
    download,
    save,
)

sys.path.insert(0, str(ROOT / "examples/training/reference/src"))
from atlas_reference.ner_model import main as train_reference
from atlas_foundations.ner_contract import LABELS, align_first, spans, entity_scores
from atlas_foundations.e7.contracts import align_all


def main():
    path = ROOT / "examples/artifacts/e7-ner/reused-training"
    # One implementation of training/selection/decoding, reused rather than copied.
    with redirect_stdout(io.StringIO()):
        result = train_reference(
            output_path=path,
            model_path=download(MODEL),
            tokenizer_path=download(TOKENIZER),
        )
    tokenizer = AutoTokenizer.from_pretrained(
        download(TOKENIZER), use_fast=True, local_files_only=True
    )
    probes = [
        dict(
            id="split-and-multiword",
            tokens=[
                "Mireille",
                "Quill",
                "visited",
                "Cedar",
                "Research",
                "in",
                "New",
                "York",
                ".",
            ],
            tags=["B-PER", "I-PER", "O", "B-ORG", "I-ORG", "O", "B-LOC", "I-LOC", "O"],
        ),
        dict(
            id="no-entities",
            tokens=["The", "doors", "are", "open", "."],
            tags=["O"] * 5,
        ),
    ]
    alignments = []
    for row in probes:
        full = tokenizer(row["tokens"], is_split_into_words=True)
        counts = Counter(i for i in full.word_ids() if i is not None)
        for cap in [32, 6]:
            encoded = tokenizer(
                row["tokens"],
                is_split_into_words=True,
                truncation=True,
                padding="max_length",
                max_length=cap,
            )
            ids = encoded.word_ids()
            truncated = Counter(i for i in ids if i is not None) != counts
            first = align_first(ids, row["tags"], counts, reject_truncated=False)
            rejected = False
            try:
                propagated = align_all(ids, row["tags"], counts)
            except ValueError:
                rejected = True
                propagated = None
            assert rejected == truncated
            for i, label in zip(ids, first):
                if i is None:
                    assert label == -100
            alignments.append(
                dict(
                    id=row["id"],
                    tokens=row["tokens"],
                    tags=row["tags"],
                    max_length=cap,
                    truncated=truncated,
                    training_rejected=rejected,
                    rows=[
                        dict(
                            position=j,
                            word_id=w,
                            word=row["tokens"][w] if w is not None else None,
                            word_tag=row["tags"][w] if w is not None else None,
                            piece=piece,
                            token_id=token,
                            label_first=a,
                            label_all=None if propagated is None else propagated[j],
                            loss_first=a != -100,
                            loss_all=None
                            if propagated is None
                            else propagated[j] != -100,
                        )
                        for j, (w, piece, token, a) in enumerate(
                            zip(
                                ids,
                                tokenizer.convert_ids_to_tokens(encoded.input_ids),
                                encoded.input_ids,
                                first,
                            )
                        )
                    ],
                )
            )
    assert any(
        sum(r["word_id"] == i for r in alignments[0]["rows"]) > 1
        for i in range(len(probes[0]["tokens"]))
    )
    gold = [["B-PER", "I-PER", "O", "B-ORG", "I-ORG", "O"]]
    errors = []
    for name, pred in [
        ("correct", gold[0]),
        ("boundary", ["B-PER", "O", "O", "B-ORG", "I-ORG", "O"]),
        ("type", ["B-LOC", "I-LOC", "O", "B-ORG", "I-ORG", "O"]),
        ("invalid-I", ["I-PER", "I-PER", "O", "B-ORG", "I-ORG", "O"]),
    ]:
        errors.append(
            dict(
                name=name,
                gold=gold[0],
                prediction=pred,
                metrics=entity_scores(gold, [pred]),
                token_accuracy=sum(a == b for a, b in zip(gold[0], pred)) / len(pred),
                raw_invalid_bio=name == "invalid-I",
                policy="Invalid I begins a new span under the shared explicit repair rule; report the raw violation separately.",
            )
        )
    output = dict(
        training=result,
        alignment=alignments,
        span_examples=errors,
        annotation_policy=json.loads(
            (ROOT / "examples/data/fixtures/ner-reference.json").read_text()
        )["policy"],
        scope="Flat nonoverlapping PER/ORG/LOC tags with disjoint name sets and shared templates; no nested entities, CoNLL performance or natural-domain generalization claim. Training uses first complete subtoken supervision. All-subtoken propagation is a separately verified alignment alternative, not a second trained model.",
        reuse="atlas_reference.ner_model.main: the original 90-step E2 trainer, validation-selected checkpoint, frozen embeddings, whole-sentence predictions and strict entity metrics. E7 changes only artifact/model paths, not a duplicated training loop.",
        reload_verified=result["reload"],
        artifact=artifact_files(path),
        verified=True,
    )
    save("e7-ner", output)
    print(
        json.dumps(
            dict(
                actual_ner_trained=True,
                alignment_verified=True,
                truncation_rejected=True,
                reload_verified=True,
            )
        )
    )


if __name__ == "__main__":
    main()
