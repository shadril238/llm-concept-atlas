relevant = {"A", "C"}
ranking = ["B", "A", "C"]
k = 2
recall_at_k = len(relevant & set(ranking[:k])) / len(relevant)
print(recall_at_k)
