values = [0.12, 0.47, 1.0]


def quantize(values, scale):
    codes = [max(-7, min(7, round(x / scale))) for x in values]
    restored = [q * scale for q in codes]
    return codes, [round(x, 3) for x in restored]


print("local range:", quantize(values, 1 / 7))
print("outlier-scaled range:", quantize(values, 1.0))
print("Uniform signed grid demonstration; not the NF4 codebook.")
