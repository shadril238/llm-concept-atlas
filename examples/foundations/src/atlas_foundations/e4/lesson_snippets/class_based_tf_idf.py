from math import log

counts = [[3, 1, 0], [0, 1, 3]]
lengths = [sum(row) for row in counts]
average = int(sum(lengths) / len(lengths))
f = [sum(row[j] for row in counts) for j in range(3)]
idf = [log(1 + average / value) for value in f]
weights = [
    [count / length * factor for count, factor in zip(row, idf)]
    for row, length in zip(counts, lengths)
]
print("A:", average, "f:", f)
print([[round(v, 6) for v in row] for row in weights])
