"""Validate an explicitly supplied export; never contacts a model provider."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'examples/foundations/src'))
from atlas_foundations.e6.external import import_grounded,import_judgments

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('kind',choices=['grounded','judge'])
 parser.add_argument('input',type=Path)
 args=parser.parse_args()
 envelope=json.loads(args.input.read_text())
 if args.kind=='grounded':
  fixture=json.loads((ROOT/'examples/data/fixtures/retrieval-corpus.json').read_text())
  result=import_grounded(fixture,envelope)
 else:
  rows=json.loads((ROOT/'src/data/retrieval-results.json').read_text())['grounded']['rows']
  result=import_judgments(rows,envelope)
 print(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False))

if __name__=='__main__':main()
