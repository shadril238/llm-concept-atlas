import json
from pathlib import Path
data = json.loads(Path('src/data/retrieval-results.json').read_text())
v = data['vision']
print(v['implementation'])
print('Manual/native preprocessing equal:',v['preprocessing_verified'])
print('Saved vectors reload checked:',v['embedding_reload_verified'])
for image in v['images']:print(image['id'],image['original_size'],'->',image['processed_shape'])
