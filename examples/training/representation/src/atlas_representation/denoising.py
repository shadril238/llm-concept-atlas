"""L17: actual tied TSDAE reconstruction, retrieval checks and supervised adaptation."""

from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atlas_representation.common import (
    ROOT,
    SentenceTransformer,
    artifact_files,
    copy,
    encoder,
    fixture,
    losses,
    np,
    pairs,
    random,
    retrieval_eval,
    save,
    seed,
    similarity_eval,
    state_hash,
    torch,
    train_embedding,
)
from atlas_foundations.e7.contracts import delete_words


def main():
    seed()
    data = fixture()
    texts = [r for r in data["rows"] if r["split"] == "train"]
    by_id = {r["id"]: r for r in texts}
    damaged = [
        dict(id=r["id"], **delete_words(r["text"], 42 + i)) for i, r in enumerate(texts)
    ]
    model = encoder("cls")
    baseline = dict(
        similarity=similarity_eval(model, pairs(data, "test")),
        retrieval=retrieval_eval(model, data, "test"),
    )
    initial = state_hash(model)
    objective = losses.DenoisingAutoEncoderLoss(model, tie_encoder_decoder=True).cpu()
    tokenizer = objective.tokenizer_decoder
    assert (
        model[0].auto_model.embeddings.word_embeddings.weight.data_ptr()
        == objective.decoder.bert.embeddings.word_embeddings.weight.data_ptr()
    )
    optimizer = torch.optim.AdamW(objective.parameters(), lr=1e-4, weight_decay=0.01)
    rng = random.Random(42)
    history = []
    gradients = {}
    best = None
    best_score = None

    def reconstruction_loss():
        objective.eval()
        values = []
        with torch.no_grad():
            for start in range(0, len(damaged), 4):
                batch = damaged[start : start + 4]
                features = [
                    model.tokenize([r[k] for r in batch])
                    for k in ["damaged", "original"]
                ]
                values.append(float(objective(features, None)))
        return sum(values) / len(values)

    decoder_initial = state_hash(objective.decoder, lambda n, p: "crossattention" in n)
    original_loss = reconstruction_loss()
    initial_retrieval = retrieval_eval(model, data, "validation")
    history.append(
        dict(
            step=0,
            reconstruction_training_probe_loss=original_loss,
            validation_mrr=initial_retrieval["mrr"],
        )
    )
    for step in range(60):
        batch = rng.sample(damaged, 4)
        objective.train()
        optimizer.zero_grad()
        features = [
            model.tokenize([r[k] for r in batch]) for k in ["damaged", "original"]
        ]
        loss = objective(features, None)
        assert torch.isfinite(loss)
        loss.backward()
        if step == 0:
            decoder_first_gradient = float(
                objective.decoder.bert.encoder.layer[
                    0
                ].crossattention.self.value.weight.grad.norm()
            )
            decoder_query_gradient = float(
                objective.decoder.bert.encoder.layer[
                    0
                ].crossattention.self.query.weight.grad.norm()
            )
            gradients = {
                n: float(p.grad.norm())
                for n, p in model.named_parameters()
                if p.grad is not None
            }
        torch.nn.utils.clip_grad_norm_(objective.parameters(), 1)
        optimizer.step()
        if (step + 1) % 10 == 0:
            value = reconstruction_loss()
            quality = retrieval_eval(model, data, "validation")
            history.append(
                dict(
                    step=step + 1,
                    reconstruction_training_probe_loss=value,
                    validation_mrr=quality["mrr"],
                )
            )
            # Match the deployed encoder's task, not the reconstruction surrogate alone.
            criterion = quality["mrr"]
            if best_score is None or criterion > best_score:
                best_score = criterion
                best = copy.deepcopy(objective.state_dict())
                best_step = step + 1
    objective.load_state_dict(best)
    assert decoder_first_gradient > 0 and decoder_initial != state_hash(
        objective.decoder, lambda n, p: "crossattention" in n
    )
    model.eval()
    assert state_hash(model) != initial and any(g > 0 for g in gradients.values())
    after_tsdae = dict(
        similarity=similarity_eval(model, pairs(data, "test")),
        retrieval=retrieval_eval(model, data, "test"),
    )
    directory = ROOT / "examples/artifacts/e7-denoising/tsdae"
    model.save_pretrained(directory)
    objective.decoder.save_pretrained(directory / "decoder")
    tokenizer.save_pretrained(directory / "decoder")
    reloaded = SentenceTransformer(str(directory), device="cpu")
    assert np.allclose(
        model.encode([r["text"] for r in texts]),
        reloaded.encode([r["text"] for r in texts]),
        atol=1e-6,
    )
    sample = damaged[0]
    objective.eval()
    with torch.no_grad():
        reps = model(model.tokenize([sample["damaged"]]))["sentence_embedding"]
        generated = objective.decoder.generate(
            input_ids=torch.tensor([[tokenizer.cls_token_id]]),
            encoder_hidden_states=reps[:, None],
            encoder_attention_mask=torch.ones((1, 1), dtype=torch.long),
            max_new_tokens=24,
            do_sample=False,
            eos_token_id=tokenizer.sep_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        raw = tokenizer.decode(generated[0], skip_special_tokens=True)
    teacher_forced = model.tokenize([sample["original"]])
    with torch.no_grad():
        ids = teacher_forced["input_ids"]
        out = objective.decoder(
            input_ids=ids[:, :-1],
            encoder_hidden_states=reps[:, None],
            encoder_attention_mask=torch.ones((1, 1), dtype=torch.long),
        )
        pred = out.logits.argmax(-1)[0].tolist()
    conditions = []
    for name, student in [("gold-only", encoder("cls")), ("tsdae-plus-gold", model)]:
        obj = losses.CosineSimilarityLoss(student)
        path = ROOT / f"examples/artifacts/e7-denoising/{name}"
        training = train_embedding(
            student, obj, data, pairs(data, "train"), steps=60, directory=path
        )
        conditions.append(
            dict(
                condition=name,
                training=training,
                similarity=similarity_eval(student, pairs(data, "test")),
                retrieval=retrieval_eval(student, data, "test"),
                artifact=artifact_files(path),
            )
        )
    counterexamples = [
        dict(before=a, after=b)
        for a, b in zip(history, history[1:])
        if b["reconstruction_training_probe_loss"]
        < a["reconstruction_training_probe_loss"]
        and b["validation_mrr"] <= a["validation_mrr"]
    ]
    save(
        "e7-denoising",
        dict(
            baseline=baseline,
            tsdae=after_tsdae,
            history=history,
            selected_step=best_step,
            selection="first maximum validation retrieval MRR, not minimum reconstruction loss",
            corruption=damaged,
            reconstruction=dict(
                source=sample,
                greedy=raw,
                greedy_ids=generated[0].tolist(),
                teacher_forced_next_tokens=tokenizer.convert_ids_to_tokens(pred),
                encoder_condition_shape=list(reps[:, None].shape),
                warning="Teacher-forced tokens see gold previous tokens; greedy reconstruction does not. Neither is evidence of retrieval improvement.",
            ),
            counterexamples=counterexamples,
            adaptation=conditions,
            initial_sha256=initial,
            first_gradient_norms=gradients,
            decoder_first_cross_attention_value_gradient=decoder_first_gradient,
            decoder_first_cross_attention_query_gradient=decoder_query_gradient,
            single_context_key_note="One encoder vector gives a one-element attention softmax: query/key selection gradients are zero; value/output paths still learn.",
            decoder_cross_attention_updated=True,
            tied_embeddings_verified=True,
            reload_verified=True,
            artifact=artifact_files(directory),
            protocol="Training-only original community-service sentences, 60% seeded whitespace-word deletion with at least one retained word, CLS pooling, one-vector decoder cross-attention and actual tied Sentence Transformers DenoisingAutoEncoderLoss. Fixed corruption is an inspectable small training probe; no general language reconstruction or benchmark claim.",
            verified=True,
        ),
    )
    print(
        json.dumps(
            dict(
                tsdae_trained=True,
                tied_parameters_verified=True,
                adaptation_compared=True,
                reload_verified=True,
                loss_retrieval_disagreements=len(counterexamples),
            )
        )
    )


if __name__ == "__main__":
    main()
