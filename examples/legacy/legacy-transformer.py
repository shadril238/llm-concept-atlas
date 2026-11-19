# Toy one-block data flow, without learned weights
states = [1.0, 2.0, 3.0]
context = sum(states) / len(states)
attention_mix = [0.7*x + 0.3*context for x in states]
feed_forward = [max(0, 2*x - 1) for x in attention_mix]
print([round(x, 2) for x in feed_forward])
