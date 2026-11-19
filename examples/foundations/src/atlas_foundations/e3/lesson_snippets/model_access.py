weights = 14_000_000
for bytes_per_value in [4, 2]:
    print(
        bytes_per_value,
        "bytes/weight:",
        round(weights * bytes_per_value / 2**20, 1),
        "MiB for weights only",
    )
