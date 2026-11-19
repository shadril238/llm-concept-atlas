vocabulary = ["cat", "dog", "bites"]
for sentence in ["dog bites cat", "cat bites dog", "dog dog"]:
    words = sentence.split()
    print(sentence, [words.count(term) for term in vocabulary])
