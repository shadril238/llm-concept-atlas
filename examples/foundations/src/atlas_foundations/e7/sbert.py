u = [1.0, 0.5]
v = [0.2, 0.8]
features = u + v + [abs(a - b) for a, b in zip(u, v)]
print('classifier features:', [round(x, 2) for x in features])
print('unordered pairs for 10,000 inputs:', 10_000 * 9_999 // 2)
print('separate encoder inputs:', 10_000)
print('scope: operation counts, not measured latency')
