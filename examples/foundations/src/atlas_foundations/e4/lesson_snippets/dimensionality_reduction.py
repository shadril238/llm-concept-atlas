from math import dist

points = [[0.0, 0.0, 0.0], [0.0, 0.0, 10.0]]
projection = [point[:2] for point in points]
print("original distance:", dist(*points))
print("display distance:", dist(*projection))
assert dist(*points) != dist(*projection)
