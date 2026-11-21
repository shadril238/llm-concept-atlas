from collections import Counter

samples = [15, 15, 14, None, None]
counts = Counter(x for x in samples if x is not None)
best = max(counts.values(), default=0)
winners = [x for x, n in counts.items() if n == best]
winner = winners[0] if len(winners) == 1 else None
print("winner:", winner, "strict majority:", best > len(samples) / 2)
print("invalid:", samples.count(None))
