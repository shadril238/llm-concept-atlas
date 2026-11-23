import sys
sys.path.insert(0, 'examples/foundations/src')
from atlas_foundations.e6.retrieval import chunks
text = 'Laptop loans last three days. Return them to the desk. Book loans last twenty-one days.'
for part in chunks(text,size=2,overlap=1,mode='sentence'):
    assert text[part['start']:part['end']]==part['text']
    print(part['start'],part['end'],part['text'])
