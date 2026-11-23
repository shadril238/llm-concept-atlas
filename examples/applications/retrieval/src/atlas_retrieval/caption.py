"""L14b: genuine BLIP-2 with a bounded CPU weight budget and disk offload."""

from pathlib import Path
import sys, json, time, hashlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atlas_retrieval.common import ROOT, snapshot, save, provenance, torch
from atlas_retrieval.vision import inverse_image
from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration

NAME = "Salesforce/blip2-opt-2.7b"


def main():
    path = snapshot(NAME)
    processor = AutoProcessor.from_pretrained(
        path, local_files_only=True, use_fast=False
    )
    offload = ROOT / ".cache/e6/blip2-offload"
    offload.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    # This caps resident model WEIGHTS, not total process RSS; one layer and runtime buffers are additional.
    model = Blip2ForConditionalGeneration.from_pretrained(
        path,
        local_files_only=True,
        use_safetensors=True,
        dtype=torch.float32,
        device_map="auto",
        max_memory={"cpu": "1GiB"},
        offload_folder=str(offload),
        offload_state_dict=True,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).eval()
    load_seconds = time.perf_counter() - started
    assert "disk" in model.hf_device_map.values(), (
        "The recorded low-memory profile must actually exercise disk offload"
    )
    trace = {}

    def hook(name):
        def record(module, inputs, output):
            value = (
                output.last_hidden_state
                if hasattr(output, "last_hidden_state")
                else output
            )
            trace[name] = list(value.shape)

        return record

    handles = [
        model.vision_model.register_forward_hook(hook("vision_tokens")),
        model.qformer.register_forward_hook(hook("query_tokens")),
        model.language_projection.register_forward_hook(hook("projected_queries")),
    ]
    rows = []
    for id, prompt, cap in [
        ("astronaut", "", 12),
        ("coffee", "Question: What is in the cup? Answer:", 12),
    ]:
        image = Image.open(ROOT / f"public/images/e6/{id}.png").convert("RGB")
        inputs = processor(images=image, text=prompt, return_tensors="pt")
        assert (
            inputs["input_ids"].dtype == torch.int64
            and inputs["pixel_values"].dtype == torch.float32
        )
        display = ROOT / f"public/images/e6/{id}-blip-input.png"
        inverse_image(
            inputs["pixel_values"][0],
            processor.image_processor.image_mean,
            processor.image_processor.image_std,
        ).save(display)
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                **inputs, do_sample=False, max_new_tokens=cap, num_beams=1
            )
        text = processor.batch_decode(output, skip_special_tokens=True)[0]
        rows.append(
            {
                "image_id": id,
                "prompt": prompt,
                "raw": text,
                "output_ids": output[0].tolist(),
                "seconds": time.perf_counter() - started,
                "max_new_tokens": cap,
                "input_shapes": {k: list(v.shape) for k, v in inputs.items()},
                "dtypes": {k: str(v.dtype) for k, v in inputs.items()},
                "input_ids": inputs["input_ids"][0].tolist(),
                "text_tokens": processor.tokenizer.convert_ids_to_tokens(
                    inputs["input_ids"][0].tolist()
                ),
                "trace": dict(trace),
                "display_input": "/images/e6/" + display.name,
                "display_sha256": hashlib.sha256(display.read_bytes()).hexdigest(),
                "pixel_sha256": hashlib.sha256(
                    inputs["pixel_values"].numpy().tobytes()
                ).hexdigest(),
            }
        )
    for handle in handles:
        handle.remove()
    assert all(
        r["trace"]["query_tokens"][1] == model.config.num_query_tokens for r in rows
    )
    result = {
        "models": provenance([NAME]),
        "profile": {
            "compute": "CPU float32",
            "weight_budget": "1 GiB",
            "offload": "remaining model weights dispatched from local safetensors/disk as needed",
            "device_map": model.hf_device_map,
            "runtime_memory_caution": "The weight-placement budget is not a total-RSS cap. Active layers, buffers, tokenizer, and page cache are additional.",
            "load_seconds": load_seconds,
        },
        "rows": rows,
        "bridge_config": {
            "query_count": model.config.num_query_tokens,
            "vision_hidden": model.config.vision_config.hidden_size,
            "query_hidden": model.config.qformer_config.hidden_size,
            "language_hidden": model.config.text_config.hidden_size,
            "vision_patch_size": model.config.vision_config.patch_size,
        },
        "actual_blip2_verified": True,
        "offload_verified": True,
        "processor_dtypes_verified": True,
        "limitations": [
            "Genuine pretrained BLIP-2 inference, not a smaller substitute or a randomly initialized bridge.",
            "Slow CPU/disk profile for a memory-limited author host. No CUDA/4-bit backend execution is claimed.",
            "Caption/VQA strings are retained unedited and can hallucinate or stop at the output cap.",
            "No BLIP-2 training performed; the original two-stage training is explained separately.",
        ],
    }
    save("e6-caption", result)
    print(
        json.dumps(
            {
                "actual_blip2_verified": True,
                "offload_verified": True,
                "processor_dtypes_verified": True,
                "cases": len(rows),
                "outputs": [r["raw"] for r in rows],
            }
        )
    )


if __name__ == "__main__":
    main()
