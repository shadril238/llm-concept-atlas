items = "the cat sleeps".split()
pairs = [
    (items[i], items[j])
    for i in range(len(items))
    for j in range(max(0, i - 1), min(len(items), i + 2))
    if i != j
]
print(pairs)
