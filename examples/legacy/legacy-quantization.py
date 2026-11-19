weights = [0.12, 0.48, -0.31]
scale = 0.1
integers = [round(weight / scale) for weight in weights]
restored = [value * scale for value in integers]
print(integers, restored)
