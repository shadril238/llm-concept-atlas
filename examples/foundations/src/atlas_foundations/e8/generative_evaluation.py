from math import exp

negative_log_likelihood, completion_tokens = 12.0, 4
print("completion NLL:", negative_log_likelihood / completion_tokens)
print(
    "completion perplexity:", round(exp(negative_log_likelihood / completion_tokens), 3)
)
reference = "The package arrived on Monday.".lower().split()
wrong = "The package arrived on Tuesday.".lower().split()
print("shared word types:", len(set(reference) & set(wrong)))
print("High lexical overlap does not verify the day.")
