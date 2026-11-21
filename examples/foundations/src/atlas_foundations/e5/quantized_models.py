file_bytes = 144811360
print("artifact MiB:", round(file_bytes / 1024**2, 2))
runtime = {"n_ctx": 768, "n_batch": 256, "n_threads": 1, "n_gpu_layers": 0}
print("declared runtime:", runtime)
assert runtime["n_batch"] <= runtime["n_ctx"]
