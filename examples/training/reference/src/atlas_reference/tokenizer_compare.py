"""Compare two actual tokenizer vocabularies and normalization policies."""

import json
from pathlib import Path
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[5]
SPECS = [
    ("google-bert/bert-base-uncased", "86b5e0934494bd15c9632b12f734a8a67f723594"),
    ("openai-community/gpt2", "607a30d783dfa663caf39e06633721c8d4cfcd7e"),
]
TEXTS = [
    "Language models learn patterns.",
    "unbelievableness",
    "안녕하세요 👋",
    'print("Hello")\n  x = 2',
]


def main():
    examples = []
    for model_id, revision in SPECS:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision, use_fast=True
        )
        for text in TEXTS:
            encoded = tokenizer(
                text, return_offsets_mapping=True, return_special_tokens_mask=True
            )
            ids = encoded["input_ids"]
            examples.append(
                {
                    "model": model_id,
                    "text": text,
                    "pieces": tokenizer.convert_ids_to_tokens(ids),
                    "ids": ids,
                    "offsets": encoded["offset_mapping"],
                    "special": encoded["special_tokens_mask"],
                    "decoded": tokenizer.decode(
                        ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    ),
                }
            )
            assert (
                len(ids)
                == len(encoded["offset_mapping"])
                == len(encoded["special_tokens_mask"])
            )
            if "gpt2" in model_id:
                assert tokenizer.decode(ids, clean_up_tokenization_spaces=False) == text
        # Empty input is still a valid encoded input; BERT adds two markers.
        assert len(tokenizer("")["input_ids"]) == (2 if "bert" in model_id else 0)
    path = ROOT / "examples/artifacts/tokenizers"
    path.mkdir(parents=True, exist_ok=True)
    (path / "results.json").write_text(
        json.dumps(examples, indent=2, ensure_ascii=False) + "\n"
    )
    print(
        json.dumps(
            {
                "tokenizers": 2,
                "cases": len(examples),
                "roundtrip_gpt2": True,
                "counts": [len(row["ids"]) for row in examples],
            }
        )
    )


# region usage
if __name__ == "__main__":
    main()
# endregion usage
