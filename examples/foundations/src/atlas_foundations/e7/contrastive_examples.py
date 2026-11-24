import sys, json
from pathlib import Path
sys.path.insert(0, 'examples/foundations/src')
from atlas_foundations.e7.contracts import check_splits, check_mnr_batch
data = json.loads(Path('examples/data/fixtures/embedding-training.json').read_text())
check_splits(data)
by_id = {r['id']: r for r in data['rows']}
positives = [p for p in data['gold_pairs'] if p['split'] == 'train' and p['label'] == 1]
batch = [next(p for p in positives if by_id[p['left_id']]['intent'] == intent)
         for intent in sorted({r['intent'] for r in data['rows']})]
print('valid MNR batch:', check_mnr_batch(batch, by_id))
try:
    check_mnr_batch([batch[0], batch[0]], by_id)
except ValueError as error:
    print('rejected:', error)
