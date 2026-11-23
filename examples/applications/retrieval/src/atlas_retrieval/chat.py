"""L14c: actual small multimodal chat, carrying the prior generated answer forward."""

from pathlib import Path
import sys, json, time, hashlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atlas_retrieval.common import ROOT, snapshot, save, provenance, torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image

NAME = "HuggingFaceTB/SmolVLM-256M-Instruct"


def main():
    path = snapshot(NAME)
    processor = AutoProcessor.from_pretrained(
        path, local_files_only=True, trust_remote_code=False
    )
    model = AutoModelForVision2Seq.from_pretrained(
        path,
        local_files_only=True,
        use_safetensors=True,
        trust_remote_code=False,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    ).eval()
    image = Image.open(ROOT / "public/images/e6/coffee.png").convert("RGB")
    conversation = []
    rows = []
    for turn, question in enumerate(
        [
            "Describe the visible objects briefly.",
            "What is the exact price of this drink? If the image does not show a price, say unknown.",
        ]
    ):
        conversation.append(
            {
                "role": "user",
                "content": ([{"type": "image"}] if turn == 0 else [])
                + [{"type": "text", "text": question}],
            }
        )
        rendered = processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        inputs = processor(
            text=rendered, images=[image], return_tensors="pt", do_image_splitting=False
        )
        assert (
            inputs["input_ids"].dtype == torch.int64
            and inputs["pixel_values"].dtype == torch.float32
        )
        count = inputs["input_ids"].shape[1]
        if count > 1536:
            raise ValueError("Declared multimodal input cap exceeded")
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=40, do_sample=False)
        continuation = output[:, count:]
        raw = processor.batch_decode(continuation, skip_special_tokens=True)[0]
        rows.append(
            {
                "turn": turn,
                "question": question,
                "messages": json.loads(json.dumps(conversation)),
                "rendered_prompt": rendered,
                "raw": raw,
                "input_shapes": {k: list(v.shape) for k, v in inputs.items()},
                "dtypes": {k: str(v.dtype) for k, v in inputs.items()},
                "input_tokens": int(count),
                "output_tokens": int(continuation.shape[1]),
                "pixel_sha256": hashlib.sha256(
                    inputs["pixel_values"].numpy().tobytes()
                ).hexdigest(),
                "seconds": time.perf_counter() - started,
                "settings": {
                    "do_sample": False,
                    "max_new_tokens": 40,
                    "image_splitting": False,
                },
            }
        )
        conversation.append(
            {"role": "assistant", "content": [{"type": "text", "text": raw}]}
        )
    assert rows[0]["raw"] in rows[1]["rendered_prompt"]
    result = {
        "models": provenance([NAME]),
        "image_id": "coffee",
        "rows": rows,
        "actual_multimodal_verified": True,
        "history_carries_generated_answer": True,
        "continuation_only_verified": True,
        "limitations": [
            "This separate 256M instruction model is a CPU-friendly chat adaptation, not BLIP-2 and not a substitute for its separately executed bridge lab.",
            "An exact price is not visible in the photograph; a confident amount would be unsupported.",
            "Two turns demonstrate application-retained history, not weight updates or general vision reliability.",
            "Original photograph can contain training overlap; not a held-out vision benchmark.",
        ],
    }
    save("e6-chat", result)
    print(
        json.dumps(
            {
                "actual_multimodal_verified": True,
                "history_carries_generated_answer": True,
                "continuation_only_verified": True,
                "turns": len(rows),
                "outputs": [r["raw"] for r in rows],
            }
        )
    )


if __name__ == "__main__":
    main()
