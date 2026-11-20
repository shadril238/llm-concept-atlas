model_labels = {0: "POSITIVE", 1: "NEGATIVE"}
application_ids = {"negative": 0, "positive": 1}
model_scores = [0.8, 0.2]
canonical = [0.0, 0.0]
for column, label in model_labels.items():
    canonical[application_ids[label.casefold()]] = model_scores[column]
print("[negative, positive]:", canonical)
assert max(range(2), key=canonical.__getitem__) == 1
