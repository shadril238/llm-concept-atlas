tokens = ['The', 'bank', 'approved', 'the', 'loan']
for i, token in enumerate(tokens):
    encoder_visible = tokens
    decoder_visible = tokens[:i+1]
    if token == 'bank':
        print('encoder:', encoder_visible)
        print('decoder:', decoder_visible)
