import json
from pathlib import Path
data = json.loads(Path('src/data/retrieval-results.json').read_text())
trace = data['vision']['patch_trace']
print('actual unfolded patches:',trace['unfolded_shape'])
print('actual projection:',trace['projection_shape'])
print('with class-token positions:',trace['positional_shape'])
assert 3*32*32==trace['unfolded_shape'][-1]
assert 7*7+1==trace['positional_shape'][0]
print('Native convolution equality:',data['vision']['patch_projection_verified'])
