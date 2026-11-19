documents = ["rain storm cloud", "sun warm beach", "cloud rain wind"]
groups = [0, 1, 0]
for group in sorted(set(groups)):
    words = [word for text, label in zip(documents, groups) if label == group for word in text.split()]
    print(group, sorted(set(words), key=lambda word: (-words.count(word), word))[:3])
