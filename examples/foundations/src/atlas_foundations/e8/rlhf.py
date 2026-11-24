epsilon, ratio = 0.2, 1.3
clipped = max(1 - epsilon, min(1 + epsilon, ratio))
for advantage in [2.0, -2.0]:
    surrogate = min(ratio * advantage, clipped * advantage)
    print("advantage", advantage, "PPO surrogate", surrogate)
print("Old rollout policy ratio differs from a fixed-reference KL penalty.")
