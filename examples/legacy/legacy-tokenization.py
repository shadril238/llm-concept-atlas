from transformers import AutoTokenizer

model_id = "google-bert/bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_id)

for text in ["Language models learn patterns.", "unbelievableness"]:
    encoded = tokenizer(text)
    ids = encoded["input_ids"]
    pieces = tokenizer.convert_ids_to_tokens(ids)

    print(f"\nInput: {text}")
    print("Pieces and IDs:", list(zip(pieces, ids)))
    print("Token count:", len(ids))
    print("Decoded:", tokenizer.decode(ids, skip_special_tokens=True))
