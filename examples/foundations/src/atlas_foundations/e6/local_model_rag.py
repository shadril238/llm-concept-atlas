import json
from pathlib import Path
data = json.loads(Path('src/data/retrieval-results.json').read_text())
print('Models and pinned revisions:')
for name,info in data['grounded']['models'].items():print(name,info['revision'])
print('Native input cap:',data['grounded']['native_input_cap'])
print('Whole-passage budget:',data['grounded']['context_budget'])
print('Index reload checked:',data['retrieval']['reload_verified'])
