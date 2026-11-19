batch, tokens, width, vocabulary = 1, 3, 4, 7
print("IDs:", (batch, tokens))
print("states:", (batch, tokens, width))
print("LM logits:", (batch, tokens, vocabulary))
