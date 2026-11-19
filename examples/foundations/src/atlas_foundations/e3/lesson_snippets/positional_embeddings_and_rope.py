from math import cos, sin, pi

a, b, angle = 1.0, 0.0, pi / 2
rotated = [a * cos(angle) - b * sin(angle), a * sin(angle) + b * cos(angle)]
print([round(x, 6) for x in rotated])
assert abs(sum(x * x for x in rotated) - 1) < 1e-12
