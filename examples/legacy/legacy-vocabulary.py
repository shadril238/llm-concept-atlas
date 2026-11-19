vocabulary = {'cat': 42, '##s': 17, '[SEP]': 2}
ids = [vocabulary[p] for p in ['cat', '##s', '[SEP]']]
reverse = {number: piece for piece, number in vocabulary.items()}
print(ids, [reverse[number] for number in ids])
