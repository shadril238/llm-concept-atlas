limit, passage, prompt, markers = 128, 90, 15, 2
remaining = limit - passage - prompt - markers
print("output budget:", remaining)
assert remaining == 21
