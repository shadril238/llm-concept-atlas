vocabulary = {"cat": 42, "##s": 17, "[SEP]": 2}
pieces = ["cat", "##s", "[SEP]"]
print("IDs:", [vocabulary[piece] for piece in pieces])
print("positions:", list(enumerate(pieces)))
