weight = 0.2
for target in [1.0, 1.0, 0.0]:
    prediction = weight
    weight -= 0.1 * 2 * (prediction - target)
print(round(weight, 3))
