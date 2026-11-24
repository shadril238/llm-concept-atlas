"""E7: numerical/data contracts and authenticity boundaries for published training evidence."""
from pathlib import Path
from collections import Counter
import copy
import hashlib
import json
import math
import sys
import unittest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'examples/foundations/src'))
from atlas_foundations.e7.contracts import cross_entropy,mnr_loss,check_splits,check_mnr_batch,delete_words,align_all
from atlas_foundations.ner_contract import align_first,entity_scores

class TrainingContracts(unittest.TestCase):
 def test_stable_cross_entropy_and_gradient_direction(self):
  self.assertAlmostEqual(cross_entropy([1004,1002],0),math.log1p(math.exp(-2)))
  self.assertAlmostEqual(cross_entropy([4,2],0),cross_entropy([1004,1002],0))
  epsilon=1e-5;grad=(cross_entropy([4+epsilon,2],0)-cross_entropy([4-epsilon,2],0))/(2*epsilon)
  self.assertAlmostEqual(grad,1/(1+math.exp(-2))-1)
  for scores,target in [([1],True),([1,2],2),([float('nan'),2],0),([],0)]:
   with self.assertRaises(ValueError):cross_entropy(scores,target)
 def test_directional_ranking_uses_every_candidate_and_requires_a_square_batch(self):
  m=mnr_loss([[.8,.4],[.3,.9]],5);self.assertAlmostEqual(m['loss'],(math.log1p(math.exp(-2))+math.log1p(math.exp(-3)))/2)
  for matrix,scale in [([[1,2,3],[2,3,4]],1),([[1]],1),([[1,2],[2,1]],0)]:
   with self.assertRaises(ValueError):mnr_loss(matrix,scale)
 def test_split_and_false_negative_guards_use_source_sentences(self):
  data=json.loads((ROOT/'examples/data/fixtures/embedding-training.json').read_text());self.assertTrue(check_splits(data));bad=copy.deepcopy(data);bad['silver_candidates'][0]['left_id']=next(r['id']for r in data['rows']if r['split']=='test')
  with self.assertRaises(ValueError):check_splits(bad)
  bad=copy.deepcopy(data);bad['rows'][1]['text']=bad['rows'][0]['text']
  with self.assertRaises(ValueError):check_splits(bad)
  by={r['id']:r for r in data['rows']};positive=[p for p in data['gold_pairs']if p['label']==1 and p['split']=='train'];batch=[next(p for p in positive if by[p['left_id']]['intent']==intent)for intent in sorted({r['intent']for r in data['rows']})];self.assertTrue(check_mnr_batch(batch,by))
  with self.assertRaises(ValueError):check_mnr_batch([batch[0],batch[0]],by)
 def test_corruption_is_seeded_and_keeps_at_least_one_word(self):
  text='Café 🐈 booking in 한글';a=delete_words(text,42,.6);self.assertEqual(a,delete_words(text,42,.6));self.assertEqual(a['damaged'],' '.join(text.split()[i]for i in a['retained_word_indices']));self.assertTrue(delete_words('one',5,.999)['damaged'])
  for text,p in [('',.5),('one',1),('one',-.1)]:
   with self.assertRaises(ValueError):delete_words(text,42,p)
 def test_real_alignment_policy_masks_specials_and_propagates_b_to_i(self):
  words=[None,0,0,1,2,None,None];tags=['B-PER','I-PER','O'];counts=Counter([0,0,1,2]);self.assertEqual(align_all(words,tags,counts),[-100,1,2,2,0,-100,-100]);self.assertEqual(align_first(words,tags,counts),[-100,1,-100,2,0,-100,-100])
  with self.assertRaises(ValueError):align_all([None,0,0,None],tags,counts)
  self.assertEqual(align_all([None,0,None],['O'],Counter([0])),[-100,0,-100])
 def test_six_notebooks_match_every_shared_training_source(self):
  p=ROOT/'examples/training/representation';e=json.loads((p/'notebook-execution.json').read_text());digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();self.assertEqual(e['lockSha256'],digest(p/'uv.lock'));self.assertEqual(e['verifierSha256'],digest(p/'verify_notebooks.py'));self.assertEqual(len(e['records']),6);self.assertGreaterEqual(len(e['sourceHashes']),11)
  for path,sha in e['sourceHashes'].items():self.assertEqual(digest(ROOT/path),sha,path)
  for r in e['records']:self.assertEqual(r['status'],'passed');self.assertEqual(r['codeCells'],4);self.assertEqual(digest(ROOT/r['path']),r['sha256'])
 def test_saved_silver_never_contains_evaluation_inputs(self):
  d=json.loads((ROOT/'src/data/training-results.json').read_text());fixture=json.loads((ROOT/'examples/data/fixtures/embedding-training.json').read_text());by={r['id']:r for r in fixture['rows']}
  for r in d['augmentation']['silver']:
   self.assertEqual(by[r['left_id']]['split'],'silver-pool');self.assertEqual(by[r['right_id']]['split'],'silver-pool');self.assertGreaterEqual(r['label'],0);self.assertLessEqual(r['label'],1);self.assertEqual(r['origin'],'trained teacher probability, not gold')
  teacher=d['augmentation']['teacher'];self.assertNotEqual(teacher['initial_sha256'],teacher['selected_sha256']);self.assertGreater(teacher['first_query_gradient_norm'],0)
 def test_actual_freezing_cpu_setfit_and_matched_sampling(self):
  d=json.loads((ROOT/'src/data/training-results.json').read_text())['classification'];self.assertEqual(len(d['runs']),18)
  for seed in [11,23,37]:
   rows=[r for r in d['runs']if r['seed']==seed];ids=set(rows[0]['train_ids']);self.assertEqual(len(ids),16)
   for r in rows:self.assertEqual(set(r['train_ids']),ids);self.assertEqual(len(r['predictions']),80)
  for r in d['runs']:
   if r['method']=='setfit':self.assertEqual(r['settings']['actual_training_device'],'cpu');self.assertNotEqual(r['initial_sha256'],r['updated_sha256']);self.assertGreater(r['first_query_gradient_norm'],0);self.assertTrue(r['reload_verified'])
   elif r['method']in ['bert-partial','bert-head-only']:
    self.assertTrue(r['frozen_verified']);self.assertEqual(r['parameters']['embeddings']['changed_tensors'],0);self.assertEqual(r['parameters']['layer_0']['changed_tensors'],0)
    if r['method']=='bert-head-only':self.assertEqual(r['parameters']['layer_1']['changed_tensors'],0)
 def test_mlm_has_real_targets_without_test_corpus_or_head_confound(self):
  d=json.loads((ROOT/'src/data/training-results.json').read_text())['mlm'];self.assertFalse(set(d['training_ids'])&set(d['test_ids']));self.assertEqual(len(d['training_ids']),200);self.assertEqual(len(d['test_ids']),80);self.assertEqual(d['smoke']['steps'],2);self.assertEqual(d['evaluated']['steps'],60);self.assertEqual(d['downstream'][0]['head_pooler_initial_sha256'],d['downstream'][1]['head_pooler_initial_sha256'])
  for r in d['mask_trace']:
   if r['special_or_padding']:self.assertEqual(r['label'],-100)
   if r['loss_included']:self.assertEqual(r['label'],r['original_id'])
 def test_tsdae_one_key_gradient_and_loss_task_disagreements_are_explicit(self):
  d=json.loads((ROOT/'src/data/training-results.json').read_text())['denoising'];self.assertEqual(d['reconstruction']['encoder_condition_shape'],[1,1,128]);self.assertTrue(d['tied_embeddings_verified']);self.assertEqual(d['decoder_first_cross_attention_query_gradient'],0);self.assertGreater(d['decoder_first_cross_attention_value_gradient'],0)
  for r in d['counterexamples']:self.assertLess(r['after']['reconstruction_training_probe_loss'],r['before']['reconstruction_training_probe_loss']);self.assertLessEqual(r['after']['validation_mrr'],r['before']['validation_mrr'])
 def test_ner_keeps_whole_sentences_and_separates_alignment_from_training(self):
  d=json.loads((ROOT/'src/data/training-results.json').read_text())['ner'];rows=d['training']['predictions'];m=entity_scores([r['gold']for r in rows],[r['predicted']for r in rows]);self.assertEqual(m,d['training']['metrics']);self.assertTrue(any(r['training_rejected']for r in d['alignment']));self.assertTrue(any(r['word_id']is None for r in d['alignment'][0]['rows']));boundary=next(r for r in d['span_examples']if r['name']=='boundary');self.assertEqual(boundary['metrics']['f1'],.5);self.assertGreater(boundary['token_accuracy'],.8)

if __name__=='__main__':unittest.main()
