gold = [0, 0, 1, 1, 1]
predicted = [0, 1, 1, -1, 1]
confusion = [[0, 0, 0], [0, 0, 0]]
for actual, prediction in zip(gold, predicted):
    confusion[actual][prediction if prediction >= 0 else 2] += 1
print("actual rows; negative, positive, invalid columns:", confusion)
print("accuracy:", sum(a == p for a, p in zip(gold, predicted)) / len(gold))
