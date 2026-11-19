weights = {"encoder": 0.5, "head": 0.2}
gradients = {"encoder": 0.4, "head": -0.3}
trainable = {"head"}
for name in trainable:
    weights[name] -= 0.1 * gradients[name]
print(weights)
