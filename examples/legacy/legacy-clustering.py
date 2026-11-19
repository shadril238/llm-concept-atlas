points = {"a": (0, 0), "b": (1, 0), "c": (8, 8)}
centers = {"near": (0, 0), "far": (8, 8)}
for name, point in points.items():
    group = min(centers, key=lambda k: sum((x-y)**2 for x,y in zip(point, centers[k])))
    print(name, group)
