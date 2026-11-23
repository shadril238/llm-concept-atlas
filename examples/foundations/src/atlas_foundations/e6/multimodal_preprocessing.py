import json
from pathlib import Path
data = json.loads(Path('src/data/retrieval-results.json').read_text())
v = data['vision']
coffee = next(r for r in v['images'] if r['id']=='coffee')
print('source size:',coffee['original_size'],'retained area:',round(coffee['area_retained'],4))
print('float tensor:',coffee['processed_shape'],coffee['dtype'])
patch = v['patch_trace']['patches'][0]
restored = [n*s+m for n,s,m in zip(patch['normalized_channel_mean'],coffee['std'],coffee['mean'])]
print('Astronaut patch mean restored to [0,1]:',[round(x,4) for x in restored])
assert all(0<=value<=1 for value in restored)
