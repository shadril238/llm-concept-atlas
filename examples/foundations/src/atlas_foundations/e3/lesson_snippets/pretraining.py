from math import log

for probability in [0.25, 0.5]:
    print("probability", probability, "loss", round(-log(probability), 3))
