"""L01: inspect a real, small pretrained decoder and its continuation."""

import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[6]
MODEL = "EleutherAI/pythia-14m"
REVISION = "cf967c0a9a04383db6f7b1108d86b2962634b4ac"


def main():
    torch.manual_seed(11)
    torch.set_num_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, attn_implementation="eager"
    ).eval()
    prompt = "The sun rises in the"
    encoded = tokenizer(prompt, return_tensors="pt")
    with torch.inference_mode():
        output = model(**encoded, output_hidden_states=True)
        generated = model.generate(
            **encoded,
            max_new_tokens=16,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    continuation = generated[0, encoded.input_ids.shape[1] :]
    assert torch.equal(generated[0, : encoded.input_ids.shape[1]], encoded.input_ids[0])
    hidden = output.hidden_states[-1]
    with torch.inference_mode():
        projected = model.get_output_embeddings()(hidden)
    assert torch.allclose(projected, output.logits, atol=1e-5)
    result = {
        "prompt": prompt,
        "input_ids": encoded.input_ids[0].tolist(),
        "pieces": tokenizer.convert_ids_to_tokens(encoded.input_ids[0].tolist()),
        "new_ids": continuation.tolist(),
        "continuation": tokenizer.decode(continuation, skip_special_tokens=True),
        "input_shape": list(encoded.input_ids.shape),
        "embedding_shape": list(model.get_input_embeddings().weight.shape),
        "hidden_shape": list(hidden.shape),
        "logits_shape": list(output.logits.shape),
        "layers": model.config.num_hidden_layers,
        "heads": model.config.num_attention_heads,
        "parameters": sum(p.numel() for p in model.parameters()),
        "prompt_preserved": True,
        "lm_head_checked": True,
    }
    path = ROOT / "examples/artifacts/e3-generation"
    path.mkdir(parents=True, exist_ok=True)
    (path / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
