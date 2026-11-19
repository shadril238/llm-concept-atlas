# A tiny context rule, not a neural encoder
sentences = ['The bank flooded by the river', 'The bank approved a loan']
for sentence in sentences:
    words = sentence.lower().split()
    sense = 'river edge' if 'river' in words else 'financial institution'
    print(sense)
