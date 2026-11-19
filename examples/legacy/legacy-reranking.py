query = {"saturday", "hours"}
docs = {"A": {"saturday", "hours", "library"}, "B": {"laptop", "loans"}}
scores = {name: len(query & words) for name, words in docs.items()}
print(sorted(scores, key=scores.get, reverse=True))
