"""L14a: actual OpenCLIP encoders, processor pixels, patch projection, and paired loss."""

from pathlib import Path
import hashlib, json, sys, time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atlas_retrieval.common import ROOT, snapshot, save, provenance, torch, np
from atlas_foundations.e6.vision import contrastive_loss
from PIL import Image
import open_clip
from torchvision import transforms

NAME = "timm/vit_base_patch32_clip_224.openai"
MEAN = [0.48145466, 0.4578275, 0.40821073]
STD = [0.26862954, 0.26130258, 0.27577711]


def inverse_image(tensor, mean=MEAN, std=STD):
    values = (
        tensor.cpu() * torch.tensor(std)[:, None, None]
        + torch.tensor(mean)[:, None, None]
    ).clamp(0, 1)
    return Image.fromarray(
        (values.permute(1, 2, 0).numpy() * 255).round().astype("uint8")
    )


def main():
    images = json.loads(
        (ROOT / "examples/data/fixtures/vision-images.json").read_text()
    )["images"]
    path = snapshot(NAME)
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained=str(path / "open_clip_model.safetensors"),
        force_quick_gelu=True,
        device="cpu",
        precision="fp32",
    )
    model.eval().requires_grad_(False)
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    manual = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    originals = [Image.open(ROOT / row["path"]).convert("RGB") for row in images]
    tensors = torch.stack([preprocess(image) for image in originals])
    assert torch.allclose(
        tensors, torch.stack([manual(image) for image in originals]), atol=1e-6
    )
    descriptions = [
        "a photograph of an astronaut in a space suit",
        "a photograph of a tabby cat",
        "a photograph of a cup of coffee on a saucer",
        "a photograph of an airplane in the sky",
        "a photograph of a red sports car",
    ]
    tokens = tokenizer(descriptions)
    started = time.perf_counter()
    with torch.inference_mode():
        image_vectors = model.encode_image(tensors, normalize=True)
        text_vectors = model.encode_text(tokens, normalize=True)
        cosine = image_vectors @ text_vectors.T
        bare = model.encode_text(
            tokenizer(["astronaut", "cat", "coffee"]), normalize=True
        )
        bare_cosine = image_vectors @ bare.T
        conv = model.visual.conv1(tensors[:1])
        patches = torch.nn.functional.unfold(
            tensors[:1], kernel_size=32, stride=32
        ).transpose(1, 2)
        projected = torch.nn.functional.linear(
            patches, model.visual.conv1.weight.reshape(768, -1)
        )
        assert torch.allclose(projected, conv.flatten(2).transpose(1, 2), atol=2e-5)
    seconds = time.perf_counter() - started
    assert torch.isfinite(cosine).all() and torch.allclose(
        image_vectors.norm(dim=-1), torch.ones(3), atol=1e-5
    )
    scale = float(model.logit_scale.exp())
    logits = (cosine[:, :3] * scale).tolist()
    loss = contrastive_loss(logits)
    labels = torch.arange(3)
    library_loss = (
        torch.nn.functional.cross_entropy(torch.tensor(logits), labels)
        + torch.nn.functional.cross_entropy(torch.tensor(logits).T, labels)
    ) / 2
    assert abs(loss["loss"] - float(library_loss)) < 1e-6
    processed = []
    for row, image, tensor in zip(images, originals, tensors):
        target = ROOT / f"public/images/e6/{row['id']}-clip-input.png"
        inverse_image(tensor).save(target)
        width, height = image.size
        short = min(width, height)
        box = [
            (width - short) / 2,
            (height - short) / 2,
            (width + short) / 2,
            (height + short) / 2,
        ]
        processed.append(
            {
                **row,
                "original_size": [width, height],
                "display_input": "/images/e6/" + target.name,
                "display_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "processed_shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "range": [float(tensor.min()), float(tensor.max())],
                "mean": MEAN,
                "std": STD,
                "source_crop_box": box,
                "area_retained": short * short / (width * height),
                "input_sha256": hashlib.sha256(tensor.numpy().tobytes()).hexdigest(),
            }
        )
    patch_rows = []
    for i in range(49):
        r, c = divmod(i, 7)
        patch = tensors[0, :, r * 32 : (r + 1) * 32, c * 32 : (c + 1) * 32]
        patch_rows.append(
            {
                "index": i,
                "row": r,
                "column": c,
                "normalized_channel_mean": patch.mean(dim=(1, 2)).tolist(),
                "projected_first_8": projected[0, i, :8].tolist(),
            }
        )
    out = ROOT / "examples/artifacts/e6-vision"
    out.mkdir(parents=True, exist_ok=True)
    np.savez(
        out / "embeddings.npz", images=image_vectors.numpy(), texts=text_vectors.numpy()
    )
    loaded = np.load(out / "embeddings.npz")
    assert np.allclose(loaded["images"] @ loaded["texts"].T, cosine.numpy(), atol=1e-6)
    result = {
        "models": provenance([NAME]),
        "implementation": "Actual open_clip_torch 3.3.0 ViT-B-32 with QuickGELU and pinned OpenAI weights, not a manually assigned feature table. The book uses Transformers CLIPModel under an OpenCLIP heading; this adaptation exercises the OpenCLIP library itself.",
        "images": processed,
        "descriptions": descriptions,
        "token_ids": tokens.tolist(),
        "cosines": cosine.tolist(),
        "bare_cosines": bare_cosine.tolist(),
        "logit_scale": scale,
        "contrastive": loss,
        "patch_trace": {
            "unfolded_shape": list(patches.shape),
            "projection_shape": list(projected.shape),
            "class_token_shape": list(model.visual.class_embedding.shape),
            "positional_shape": list(model.visual.positional_embedding.shape),
            "patches": patch_rows,
        },
        "embedding_shapes": {
            "image": list(image_vectors.shape),
            "text": list(text_vectors.shape),
        },
        "preprocessing_verified": True,
        "patch_projection_verified": True,
        "embedding_reload_verified": True,
        "contrastive_library_verified": True,
        "seconds": seconds,
        "limitations": [
            "Three familiar public-domain/CC0 sample photographs; likely pretraining overlap is unknown, so this is not a held-out vision benchmark.",
            "Cosine and candidate-normalized softmax are not calibrated object-presence probabilities.",
            "The paired loss is computed from frozen real embeddings; no CLIP training or quality improvement is claimed.",
            "Inverse-normalized display approximates the actual float pixels within rounding; original normalized values and hashes are retained.",
        ],
    }
    save("e6-vision", result)
    print(
        json.dumps(
            {
                k: result[k]
                for k in [
                    "preprocessing_verified",
                    "patch_projection_verified",
                    "embedding_reload_verified",
                    "contrastive_library_verified",
                    "embedding_shapes",
                ]
            }
        )
    )


if __name__ == "__main__":
    main()
