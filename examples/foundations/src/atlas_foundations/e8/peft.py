layers, d_in, d_out, rank = 6, 128, 384, 4
base_parameters = 14_067_712
trainable = layers * rank * (d_in + d_out)
print("trainable:", trainable)
print("fraction:", round(trainable / (base_parameters + trainable), 6))
print("adapter gradients + Adam bytes:", trainable * (4 + 8))
print("Frozen backbone and activations still occupy memory.")
