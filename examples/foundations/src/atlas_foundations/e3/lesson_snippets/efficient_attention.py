layers, positions, width, bytes_per_value = 4, 100, 8, 2
for kv_heads in [4, 2, 1]:
    size = 2 * layers * kv_heads * positions * width * bytes_per_value
    print("KV heads:", kv_heads, "cache bytes:", size)
