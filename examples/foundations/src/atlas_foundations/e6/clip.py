import json
from pathlib import Path
data = json.loads(Path('src/data/retrieval-results.json').read_text())
import sys
sys.path.insert(0, 'examples/foundations/src')
from atlas_foundations.e6.vision import contrastive_loss
v = data['vision']
logits = [[cosine*v['logit_scale'] for cosine in row[:3]] for row in v['cosines']]
loss = contrastive_loss(logits)
assert abs(loss['loss']-v['contrastive']['loss'])<1e-10
print({key:round(loss[key],6) for key in ['image_to_text','text_to_image','loss']})
print('No weights were updated.')
