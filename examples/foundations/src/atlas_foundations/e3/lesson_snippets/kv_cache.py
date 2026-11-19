layers, batch, kv_heads, positions, head_width, bytes_per_value = 4, 1, 2, 10, 8, 2
size = 2 * layers * batch * kv_heads * positions * head_width * bytes_per_value
print("cache bytes:", size)
print("doubling positions:", size * 2)
