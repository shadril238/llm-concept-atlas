from fractions import Fraction

answer = Fraction(8) / (Fraction(3) - Fraction(8, 3))
print("8 / (3 - 8/3) =", answer)
assert answer == 24
print("inputs used: 8, 3, 8, 3")
