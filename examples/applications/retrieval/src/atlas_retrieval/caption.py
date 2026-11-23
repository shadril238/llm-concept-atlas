"""L14b: genuine BLIP-2 in a measured Linux CPU bfloat16 inference profile."""

from pathlib import Path
import sys, json, time, hashlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atlas_retrieval.common import ROOT, snapshot, save, provenance, torch
from atlas_retrieval.vision import inverse_image
from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration

NAME = "Salesforce/blip2-opt-2.7b"


def main():
    import psutil

    if sys.platform != "linux":
        raise RuntimeError(
            "This heavy BLIP-2 profile is verified on Linux CPU. Use the recorded result or the opt-in CI job; the small chat/OpenCLIP profiles run on the author Mac."
        )
    available = psutil.virtual_memory().available
    if available < 10 * 1024**3:
        raise RuntimeError(
            "This full-model CPU profile requires at least 10 GiB available before loading; use recorded output when the profile does not fit."
        )
    path = snapshot(NAME)
    processor = AutoProcessor.from_pretrained(
        path, local_files_only=True, use_fast=False
    )
    started = time.perf_counter()
    # The heavy verified profile runs on Linux CI, not the memory-constrained author Mac.
    model = Blip2ForConditionalGeneration.from_pretrained(
        path,
        local_files_only=True,
        use_safetensors=True,
        dtype=torch.bfloat16,
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).eval()
    load_seconds = time.perf_counter() - started
    assert all(
        module.weight.ndim == 2
        for module in model.modules()
        if isinstance(module, torch.nn.Linear)
    )
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())
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
        with (
            torch.inference_mode(),
            torch.autocast("cpu", dtype=torch.bfloat16, cache_enabled=False),
        ):
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
            "compute": "Linux CPU bfloat16 autocast; checkpoint weights bfloat16 with Q-Former/query parameters retained in float32 by Transformers",
            "weight_budget": "Full resident 16-bit model weights plus float32 Q-Former and runtime buffers; measured process memory is in the execution record",
            "device_map": model.hf_device_map,
            "runtime_memory_caution": "This heavy profile requires substantially more memory than the small local labs. It is verified on Linux CI; CUDA and the failed disk-offload profile are not claimed verified.",
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
        "cpu_profile_verified": True,
        "processor_dtypes_verified": True,
        "limitations": [
            "Genuine pretrained BLIP-2 inference, not a smaller substitute or a randomly initialized bridge.",
            "Heavy Linux CPU bfloat16 profile. No CUDA, 4-bit, or successful disk-offload execution is claimed.",
            "Caption/VQA strings are retained unedited and can hallucinate or stop at the output cap.",
            "No BLIP-2 training performed; the original two-stage training is explained separately.",
        ],
    }
    save("e6-caption", result)
    print(
        json.dumps(
            {
                "actual_blip2_verified": True,
                "cpu_profile_verified": True,
                "processor_dtypes_verified": True,
                "cases": len(rows),
                "outputs": [r["raw"] for r in rows],
            }
        )
    )


if __name__ == "__main__":
    main()
