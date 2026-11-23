import json
from pathlib import Path
data = json.loads(Path('src/data/retrieval-results.json').read_text())
matrix = data['vision']['cosines']
for image,row in zip(data['vision']['images'],matrix):
    best=max(range(len(row)),key=lambda i:row[i])
    print(image['id'],'->',data['vision']['descriptions'][best],round(row[best],4))
print('shape:',data['vision']['embedding_shapes'])
print('These are actual model similarities, not manually assigned semantic vectors.')
