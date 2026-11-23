"""E6 contracts: ranking denominators, source spans, semantic boundaries and real evidence."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'examples/foundations/src'))
from atlas_foundations.e6.retrieval import metrics,chunks,collapse,bm25,citation_metrics
from atlas_foundations.e6.vision import contrastive_loss
from atlas_foundations.e6.external import import_grounded,import_judgments

class RetrievalContracts(unittest.TestCase):
 def test_ap_uses_all_relevant_documents_and_undefined_absence(self):
  m=metrics(['a','x','b'],{'a':2,'b':1,'c':2,'d':1},3)
  self.assertAlmostEqual(m['ap'],(1+2/3)/4)
  self.assertEqual(m['precision'],2/3);self.assertEqual(m['recall'],.5)
  self.assertEqual(metrics(['a'],{'a':1},3)['precision'],1/3)
  for key in ['ap','recall','rr','ndcg']:self.assertIsNone(metrics(['a'],{},3)[key])
  for args in [(['a','a'],{'a':1},3),(['a'],{'a':True},3),(['a'],{'a':-1},3),(['a'],{'a':1},True)]:
   with self.assertRaises(ValueError):metrics(*args)
  self.assertEqual(collapse([{'document_id':'a'},{'document_id':'a'},{'document_id':'b'}]),['a','b'])
 def test_chunks_preserve_unicode_source_offsets_and_overlap(self):
  text='  Café 🐈 sleeps.\n한글 follows! Last sentence?  '
  out=chunks(text,2,1)
  self.assertEqual(len(out),2)
  for row in out:self.assertEqual(text[row['start']:row['end']],row['text'])
  self.assertIn('한글 follows!',out[0]['text']);self.assertTrue(out[1]['text'].startswith('한글'))
  self.assertEqual(chunks('',2,1),[])
  self.assertEqual(chunks('single',2,1)[0]['end'],6)
  for size,overlap,mode in [(0,0,'word'),(2,2,'word'),(True,0,'word'),(2,1,'unknown')]:
   with self.assertRaises(ValueError):chunks(text,size,overlap,mode)
 def test_bm25_handles_empty_documents_without_inventing_a_hit(self):
  scores=bm25('Birch',['','Birch access code','Wren room'],b=1)
  self.assertEqual(scores[0],0);self.assertGreater(scores[1],scores[2])
  self.assertEqual(bm25('absent',['one','two']),[0,0])
  with self.assertRaises(ValueError):bm25('x',['',''])
 def test_citation_recall_needs_joint_support_not_a_partial_link(self):
  partial={'fully_supported_by_citations':False,'citations':[{'supports':True}]}
  full={'fully_supported_by_citations':True,'citations':[{'supports':True},{'supports':True}]}
  m=citation_metrics([partial,full]);self.assertEqual(m['citation_precision'],1);self.assertEqual(m['citation_recall'],.5)
  self.assertIsNone(citation_metrics([])['citation_recall'])
  with self.assertRaises(ValueError):citation_metrics([{'fully_supported_by_citations':True,'citations':[]}])
 def test_offline_provider_import_keeps_missing_errors_and_exact_spans(self):
  f={'queries':[{'id':'q','split':'evaluation'},{'id':'missing','split':'evaluation'},{'id':'failed','split':'evaluation'},{'id':'d','split':'design'}],'documents':[{'id':'s','text':'Café 🐈 closes.'}]}
  e=dict(provider='manual-export',model='example-version',prompt_version='v1',collected_at='2026-11-24',outputs=[dict(id='q',answer='🐈 closes.',citations=[dict(source_id='s',answer_start=0,answer_end=1,source_start=5,source_end=6,quote='🐈')]),dict(id='failed',answer=None,error='request failed')])
  r=import_grounded(f,e);self.assertEqual(r['missing_ids'],['missing']);self.assertEqual(r['errors'],1);self.assertEqual(r['requested'],3);self.assertEqual(r['outputs']['q']['citations'][0]['semantic_support'],'not judged')
  for change in ['span','quote','duplicate','design']:
   bad=copy.deepcopy(e)
   if change=='span':bad['outputs'][0]['citations'][0]['answer_end']=True
   elif change=='quote':bad['outputs'][0]['citations'][0]['quote']='cat'
   elif change=='duplicate':bad['outputs'].append(bad['outputs'][0])
   else:bad['outputs'][0]['id']='d'
   with self.assertRaises(ValueError):import_grounded(f,bad)
 def test_optional_judge_is_bound_to_the_raw_draft(self):
  rows=[dict(id='q',variant='baseline',answer_call={'raw':'unknown'})]
  e=dict(judge_model='external-export',rubric_version='v1',rows=[dict(id='q',variant='baseline',raw_sha256=hashlib.sha256(b'unknown').hexdigest(),support='uncertain',fluency=2)])
  self.assertEqual(import_judgments(rows,e)['received'],1)
  with self.assertRaises(ValueError):import_judgments([dict(id='q',variant='baseline',answer_call={'raw':'edited'})],e)
 def test_notebook_evidence_covers_five_actual_fresh_kernels(self):
  project=ROOT/'examples/applications/retrieval';seen=set()
  digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
  for name,count in [('local',4),('blip2',1)]:
   e=json.loads((project/f'notebook-execution-{name}.json').read_text());self.assertEqual(e['lockSha256'],digest(project/'uv.lock'));self.assertEqual(e['verifierSha256'],digest(project/'verify_notebooks.py'));self.assertEqual(len(e['records']),count)
   for r in e['records']:
    seen.add(r['path']);self.assertEqual(r['status'],'passed');self.assertEqual(r['codeCells'],4);self.assertEqual(r['sha256'],digest(ROOT/r['path']))
  self.assertEqual(len(seen),5)
 def test_author_reviews_target_actual_raw_outputs_and_image_previews(self):
  d=json.loads((ROOT/'src/data/retrieval-results.json').read_text());reviews=d['grounded']['annotations']['rows'];self.assertEqual(len(reviews),48)
  for r in d['grounded']['rows']:
   a=next(x for x in reviews if (x['id'],x['variant'])==(r['id'],r['variant']));self.assertEqual(a['raw_sha256'],hashlib.sha256(r['answer_call']['raw'].encode()).hexdigest());self.assertLessEqual(r['search_calls'],3);self.assertLessEqual(r['answer_call']['prompt_tokens'],384)
  for r in d['vision']['images']+d['caption']['rows']:
   self.assertEqual(hashlib.sha256((ROOT/'public'/r['display_input'].lstrip('/')).read_bytes()).hexdigest(),r['display_sha256'])
  self.assertEqual(d['caption']['rows'][0]['trace']['query_tokens'],[1,32,768]);self.assertEqual(d['caption']['rows'][0]['dtypes']['input_ids'],'torch.int64')
  for condition in ['single-image','split-images']:
   first,second=[r for r in d['chat']['rows']if r['condition']==condition];self.assertIn(first['raw'],second['rendered_prompt']);self.assertEqual(second['messages'][-2]['role'],'assistant')

if __name__=='__main__':unittest.main()
