requested_quantity, extracted_quantity, price = 3, 8, 2
calculated = extracted_quantity * price
copied = calculated
print("arithmetic for extraction:", calculated)
print("copy consistent:", copied == calculated)
print("end-to-end correct:", copied == requested_quantity * price)
