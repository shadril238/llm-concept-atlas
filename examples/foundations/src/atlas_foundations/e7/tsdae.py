import sys
sys.path.insert(0, 'examples/foundations/src')
from atlas_foundations.e7.contracts import delete_words
result = delete_words('How do I book a workshop room?', seed=42, probability=0.6)
print('original:', result['original'])
print('damaged:', result['damaged'])
print('retained positions:', result['retained_word_indices'])
print('This corruption calculation does not reconstruct or train a neural model.')
