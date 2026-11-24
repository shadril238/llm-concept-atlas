weights, block_size = 1024, 64
payload = weights * 4 // 8
blocks = (weights + block_size - 1) // block_size
scales = blocks * 4
print("ideal four-bit weight bytes:", payload)
print("uncompressed float32 block-scale bytes:", scales)
print("subtotal:", payload + scales)
print("This arithmetic does not execute NF4, gradients, or a paged optimizer.")
