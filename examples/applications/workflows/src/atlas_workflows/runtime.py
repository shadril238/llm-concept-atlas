"""A pinned local GGUF model, explicit ChatML rendering, and native grammar."""

import hashlib
import json
import os
from pathlib import Path
import sys
import time
import urllib.request

# These educational workflows have no hosted provider or tracing service.
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "examples/foundations/src"))
REPOSITORY = "bartowski/SmolLM2-135M-Instruct-GGUF"
REVISION = "09816acd5d99df7be770d85ea30822623dab342c"
FILENAME = "SmolLM2-135M-Instruct-Q8_0.gguf"
SHA256 = "5a1395716f7913741cc51d98581b9b1228d80987a9f7d3664106742eb06bba83"


def fixture():
    return json.loads((ROOT / "examples/data/fixtures/workflow-tasks.json").read_text())


def save(name, result):
    path = ROOT / "examples/artifacts" / name
    path.mkdir(parents=True, exist_ok=True)
    (path / "results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )


def render(messages):
    if not messages or any(
        set(m) != {"role", "content"}
        or m["role"] not in ["system", "user", "assistant"]
        or not isinstance(m["content"], str)
        for m in messages
    ):
        raise ValueError("Explicit supported roles and string content are required")
    if any(
        marker in message["content"]
        for message in messages
        for marker in ["<|im_start|>", "<|im_end|>"]
    ):
        raise ValueError(
            "Literal role delimiters are not accepted inside this lab's data"
        )
    return (
        "".join(
            f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
            for message in messages
        )
        + "<|im_start|>assistant\n"
    )


class LocalModel:
    def __init__(self):
        from llama_cpp import Llama

        path = ROOT / ".cache/e5" / FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            if os.getenv("HF_HUB_OFFLINE") == "1":
                raise ValueError(
                    "GGUF cache missing; use --allow-network for the initial download"
                )
            urllib.request.urlretrieve(
                f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{FILENAME}",
                path,
            )
        assert hashlib.sha256(path.read_bytes()).hexdigest() == SHA256
        assert path.read_bytes()[:4] == b"GGUF"
        # region loading
        self.model = Llama(
            model_path=str(path),
            n_ctx=768,
            n_batch=256,
            n_threads=1,
            n_threads_batch=1,
            n_gpu_layers=0,
            offload_kqv=False,
            op_offload=False,
            seed=37,
            verbose=False,
        )
        # endregion loading
        self.calls = []
        self.metadata = {
            "repository": REPOSITORY,
            "revision": REVISION,
            "filename": FILENAME,
            "sha256": SHA256,
            "file_bytes": path.stat().st_size,
            "general_file_type": self.model.metadata.get("general.file_type"),
            "architecture": self.model.metadata.get("general.architecture"),
            "training_context": self.model.metadata.get("llama.context_length"),
            "chat_template": self.model.metadata.get("tokenizer.chat_template"),
            "runtime": {
                "n_ctx": 768,
                "n_batch": 256,
                "n_threads": 1,
                "n_threads_batch": 1,
                "n_gpu_layers": 0,
                "offload_kqv": False,
                "op_offload": False,
                "device": "cpu",
            },
        }

    def count(self, messages):
        return len(
            self.model.tokenize(render(messages).encode(), add_bos=False, special=True)
        )

    def generate(self, messages, max_tokens=64, temperature=0.0, seed=37, schema=None):
        from llama_cpp import LlamaGrammar

        prompt = render(messages)
        ids = self.model.tokenize(prompt.encode(), add_bos=False, special=True)
        if len(ids) + max_tokens > 768:
            raise ValueError(
                "Prompt plus reserved output exceeds this runtime's context cap"
            )
        grammar = (
            LlamaGrammar.from_json_schema(json.dumps(schema), verbose=False)
            if schema
            else None
        )
        start = time.perf_counter()
        output = self.model.create_completion(
            prompt=ids,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9 if temperature > 0 else 1.0,
            top_k=40,
            seed=seed,
            grammar=grammar,
            stop=["<|im_end|>", "<|im_start|>"],
            echo=False,
        )
        elapsed = time.perf_counter() - start
        assert output["usage"]["prompt_tokens"] == len(ids)
        call = {
            "messages": messages,
            "rendered_prompt": prompt,
            "prompt_tokens": len(ids),
            "output_tokens": output["usage"]["completion_tokens"],
            "raw": output["choices"][0]["text"],
            "finish_reason": output["choices"][0]["finish_reason"],
            "seconds": elapsed,
            "settings": {
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 0.9 if temperature > 0 else 1.0,
                "top_k": 40,
                "seed": seed,
                "native_schema": schema,
            },
        }
        self.calls.append(call)
        return call


def messages(
    user,
    system="Follow the stated task and output contract. Do not invent missing facts.",
):
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
