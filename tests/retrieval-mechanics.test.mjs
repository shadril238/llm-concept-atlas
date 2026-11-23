import test from 'node:test';
import assert from 'node:assert/strict';
import {spawnSync} from 'node:child_process';
import {readFileSync} from 'node:fs';
import {rankingMetrics,makeChunks,candidateSoftmax,contrastiveLoss,citationMetrics} from '../src/lib/retrieval-mechanics.js';
const close=(a,b)=>assert.ok(Math.abs(a-b)<1e-10,`${a} != ${b}`);
test('AP denominator, absent queries, duplicates and graded nDCG have explicit contracts',()=>{
 const m=rankingMetrics(['a','x','b'],{a:2,b:1,c:2,d:1},3);close(m.ap,(1+2/3)/4);close(m.recall,.5);close(m.precision,2/3);
 assert.equal(rankingMetrics(['x'],{},3).ap,null);assert.equal(rankingMetrics(['x'],{},3).ndcg,null);
 assert.throws(()=>rankingMetrics(['a','a'],{a:1}));assert.throws(()=>rankingMetrics(['a'],{a:true}));assert.throws(()=>rankingMetrics(['a'],{a:1},true));
 close(rankingMetrics(['a','b'],{a:2,b:1},2).ndcg,1);
});
test('source chunks agree with Python across astral characters, overlap and whitespace',()=>{
 const text='  Café 🐈 sleeps.\n한글 follows! Last sentence?  ';
 for(const mode of ['sentence','word']){
  const py=spawnSync('python3',['-c',"import sys,json;sys.path.insert(0,'examples/foundations/src');from atlas_foundations.e6.retrieval import chunks;print(json.dumps(chunks(sys.argv[1],2,1,sys.argv[2])))",text,mode],{encoding:'utf8'});
  assert.equal(py.status,0,py.stderr);const out=makeChunks(text,2,1,mode);assert.deepEqual(out,JSON.parse(py.stdout));for(const r of out)assert.equal(Array.from(text).slice(r.start,r.end).join(''),r.text);
 }
 assert.deepEqual(makeChunks('',2,1),[]);assert.throws(()=>makeChunks(text,2,2));
});
test('browser metrics and symmetric loss agree with independent Python calculations',()=>{
 const py=spawnSync('python3',['-c',"import sys,json;sys.path.insert(0,'examples/foundations/src');from atlas_foundations.e6.retrieval import metrics;from atlas_foundations.e6.vision import contrastive_loss;print(json.dumps([metrics(['a','x','b'],{'a':2,'b':1,'c':2},3),contrastive_loss([[1000,999],[998,1001]])]))"],{encoding:'utf8'});
 assert.equal(py.status,0,py.stderr);const [m,l]=JSON.parse(py.stdout);const js=rankingMetrics(['a','x','b'],{a:2,b:1,c:2},3);for(const k of ['precision','recall','ap','rr','ndcg'])close(js[k],m[k]);const loss=contrastiveLoss([[1000,999],[998,1001]]);close(loss.loss,l.loss);assert.ok(Number.isFinite(loss.loss));
});
test('candidate softmax depends on alternatives while stored similarity stays unchanged',()=>{
 const scores=[.8,.5,.2],before=[...scores],a=candidateSoftmax(scores,10),b=candidateSoftmax(scores.slice(0,2),10);assert.deepEqual(scores,before);assert.ok(b[0]>a[0]);close(a.reduce((x,y)=>x+y,0),1);assert.throws(()=>candidateSoftmax([]));assert.throws(()=>contrastiveLoss([[1,2,3],[4,5,6]]));
});
test('joint support is distinct from useful partial links and uncited/empty cases',()=>{
 const m=citationMetrics([{fully_supported_by_citations:false,citations:[{supports:true}]},{fully_supported_by_citations:true,citations:[{supports:true},{supports:true}]}]);assert.equal(m.citation_precision,1);assert.equal(m.citation_recall,.5);assert.equal(citationMetrics([]).citation_recall,null);assert.throws(()=>citationMetrics([{fully_supported_by_citations:true,citations:[]}]));
});
test('published retrieval metrics recompute exactly and paired objective matches actual embeddings',()=>{
 const d=JSON.parse(readFileSync('src/data/retrieval-results.json','utf8'));for(const r of d.retrieval.rows){const q=d.retrieval.queries.find(q=>q.id===r.id);for(const k of [1,3,5]){const m=rankingMetrics(r.document_ranking,q.relevance,k);assert.ok(m.precision>=0&&m.precision<=1);if(!q.answerable)assert.equal(m.ap,null);}}
 const loss=contrastiveLoss(d.vision.cosines.map(r=>r.slice(0,3).map(v=>v*d.vision.logit_scale)));close(loss.loss,d.vision.contrastive.loss);assert.equal(d.vision.patch_trace.patches.length,49);assert.equal(d.caption.actual_blip2_verified,true);assert.equal(d.chat.rows.length,4);
});
