"""Run with the locked generation Python after downloading the pinned base.

Compare independently flattened shifted cross entropy with completion log sums;
unequal lengths and shared EOS/pad IDs expose masking/weighting regressions.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from atlas_generation.common import base, tokenizer, batch, logps, torch, F


def main():
    tok = tokenizer()
    model = base().eval()
    rows = [
        dict(
            id="a", instruction="Return code.", context="Code: LARK.", response="LARK"
        ),
        dict(
            id="b",
            instruction="Return description.",
            context="Room: blue and quiet.",
            response="blue and quiet",
        ),
    ]
    b = batch(tok, rows)
    with torch.no_grad():
        logits = model(
            input_ids=b["input_ids"], attention_mask=b["attention_mask"]
        ).logits
        manual = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            b["labels"][:, 1:].reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        values, counts = logps(model, b)
        assert abs(manual.item() + values.sum().item()) < 1e-4
        builtin = model(**b).loss
        assert abs(builtin.item() - manual.item() / counts.sum().item()) < 1e-5
    for i, r in enumerate(rows):
        true_length = int(b["attention_mask"][i].sum())
        assert b["input_ids"][i, true_length - 1] == tok.eos_token_id
        assert b["labels"][i, true_length - 1] == tok.eos_token_id
        assert (b["labels"][i, true_length:] == -100).all()
    print(
        "Real-model causal shift, completion aggregation, EOS and unequal-length padding verified."
    )


if __name__ == "__main__":
    main()
