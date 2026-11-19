from collections import Counter
sentence = "red fox meets red bird".split()
counts = Counter(sentence)
print(dict(counts))
