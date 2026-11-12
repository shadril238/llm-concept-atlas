import { useState } from 'react';

const vectorPoints = [
  { label: 'cat', x: 0.86, y: 0.48 },
  { label: 'kitten', x: 0.78, y: 0.44 },
  { label: 'dog', x: 0.65, y: 0.75 },
  { label: 'puppy', x: 0.61, y: 0.72 },
  { label: 'car', x: -0.67, y: -0.32 },
  { label: 'truck', x: -0.81, y: -0.27 },
];

function cosine(a: { x: number; y: number }, b: { x: number; y: number }) {
  return (a.x * b.x + a.y * b.y) / (Math.hypot(a.x, a.y) * Math.hypot(b.x, b.y));
}

export function EmbeddingLab() {
  const [anchor, setAnchor] = useState('cat');
  const selected = vectorPoints.find((point) => point.label === anchor)!;
  const sorted = vectorPoints.filter((point) => point.label !== anchor)
    .map((point) => ({ ...point, score: cosine(selected, point) }))
    .sort((a, b) => b.score - a.score);
  return <section className="learning-lab" aria-label="Embedding similarity simulation">
    <div className="lab-heading"><div><span className="eyebrow">Interactive lab · illustrative vectors</span><h3>Find nearby meanings</h3></div><button type="button" onClick={() => setAnchor('cat')}>Reset</button></div>
    <p>Choose an anchor. The arrows are invented two-dimensional vectors; the ranking uses cosine similarity of their directions. Real text embeddings usually have many more dimensions and are learned from data.</p>
    <label>Anchor word <select value={anchor} onChange={(event) => setAnchor(event.target.value)}>{vectorPoints.map((point) => <option key={point.label}>{point.label}</option>)}</select></label>
    <div className="lab-grid"><svg className="vector-svg" viewBox="0 0 400 320" role="img" aria-label={`Vector arrows from an origin; ${anchor} is highlighted`}>
      <line x1="30" y1="160" x2="370" y2="160" className="axis"/><line x1="200" y1="20" x2="200" y2="300" className="axis"/>
      <text x="206" y="178" className="svg-label">origin</text>
      {vectorPoints.map((point) => { const x=200+point.x*160, y=160-point.y*135; return <g key={point.label}><line x1="200" y1="160" x2={x} y2={y} className={point.label===anchor?'vector selected':'vector'}/><circle cx={x} cy={y} r="5" className={point.label===anchor?'dot selected':'dot'}/><text x={x+7} y={y-7} className="svg-label">{point.label}</text></g>; })}
    </svg><div><h4>Closest to “{anchor}”</h4><ol className="score-list">{sorted.map((point) => <li key={point.label}><span>{point.label}</span><meter min="-1" max="1" value={point.score} aria-label={`${point.label} cosine score ${point.score.toFixed(2)}`}/><strong>{point.score.toFixed(2)}</strong></li>)}</ol></div></div>
    <p className="lab-takeaway"><strong>What to notice:</strong> cosine compares direction, not exact wording or factual correctness. This picture is a teaching projection, not measured model output.</p>
  </section>;
}

const words = ['The', 'cat', 'chased', 'mice'];
const queries = [[1, 0], [0.9, 0.2], [0.3, 1], [0.65, 0.9]];
const keys = [[0.8, 0.1], [1, 0.2], [0.3, 0.95], [0.7, 0.8]];
const values = [[1, 0], [0.8, 0.3], [0.2, 1], [0.6, 0.8]];

export function AttentionLab() {
  const [position, setPosition] = useState(2);
  const [causal, setCausal] = useState(true);
  const scores = keys.map((key, i) => causal && i > position ? null : (queries[position][0]*key[0]+queries[position][1]*key[1])/Math.sqrt(2));
  const exponentials = scores.map((score) => score === null ? 0 : Math.exp(score));
  const total = exponentials.reduce((sum, value) => sum + value, 0);
  const weights = exponentials.map((value) => value / total);
  const output = [0,1].map((dimension) => weights.reduce((sum, weight, i) => sum + weight * values[i][dimension], 0));
  return <section className="learning-lab" aria-label="Attention simulation">
    <div className="lab-heading"><div><span className="eyebrow">Interactive lab · one toy attention head</span><h3>Who contributes to this token?</h3></div><button type="button" onClick={() => {setPosition(2);setCausal(true);}}>Reset</button></div>
    <p>Select a token, then compare its query with each available key. Softmax turns the scores into weights. The final vector is the weighted sum of values.</p>
    <div className="lab-controls"><label>Token to update <select value={position} onChange={(event) => setPosition(Number(event.target.value))}>{words.map((word,i) => <option value={i} key={i}>{i+1}. {word}</option>)}</select></label><label><input type="checkbox" checked={causal} onChange={(event) => setCausal(event.target.checked)}/> Hide future tokens (decoder mask)</label></div>
    <div className="attention-rows" role="list" aria-label="Attention weights">{words.map((word,i) => <div className="attention-row" role="listitem" key={i}><span>{word}</span><span className="attention-track"><span style={{width:`${(weights[i]*100).toFixed(2)}%`}}/></span><strong>{scores[i] === null ? 'masked' : `${(weights[i]*100).toFixed(1)}%`}</strong></div>)}</div>
    <p className="lab-equation">Updated vector = {words.map((word,i) => `${weights[i].toFixed(2)} × value(${word})`).join(' + ')} = [{output.map((value) => value.toFixed(2)).join(', ')}]</p>
    <p className="lab-takeaway"><strong>What to notice:</strong> masking removes future positions before normalization. These weights are invented for learning; a real transformer has learned projections, many heads, and many layers. A weight alone is not a full explanation of a prediction.</p>
  </section>;
}

const ragExamples = [
  { question: 'When does the library close on Saturday?', terms: ['library','saturday','close'], answer: 'The library closes at 5 p.m. on Saturday.' },
  { question: 'Can I borrow a laptop?', terms: ['borrow','laptop'], answer: 'Yes. Members can borrow a laptop for up to three days.' },
];
const ragDocs = [
  { id: 'A', title: 'Library hours', text: 'The library closes at 5 p.m. on Saturday and 7 p.m. on weekdays.', terms: ['library','saturday','close'] },
  { id: 'B', title: 'Laptop loans', text: 'Members can borrow a laptop for up to three days.', terms: ['borrow','laptop'] },
  { id: 'C', title: 'Café menu', text: 'The café serves tea, coffee, and snacks.', terms: ['café','coffee'] },
];

export function RagLab() {
  const [questionIndex, setQuestionIndex] = useState(0);
  const [selected, setSelected] = useState<string[]>(['A']);
  const example = ragExamples[questionIndex];
  const ranked = ragDocs.map((doc) => ({...doc, score: example.terms.filter((term) => doc.terms.includes(term)).length})).sort((a,b) => b.score-a.score);
  const needed = questionIndex === 0 ? 'A' : 'B';
  const answer = selected.includes(needed) ? example.answer + ` [${needed}]` : 'I cannot answer from the selected passages.';
  return <section className="learning-lab" aria-label="RAG pipeline simulation">
    <div className="lab-heading"><div><span className="eyebrow">Interactive lab · deterministic example</span><h3>Build an answer from evidence</h3></div><button type="button" onClick={() => {setQuestionIndex(0);setSelected(['A']);}}>Reset</button></div>
    <label>Question <select value={questionIndex} onChange={(event) => {const index=Number(event.target.value);setQuestionIndex(index);setSelected([index===0?'A':'B']);}}>{ragExamples.map((item,i) => <option value={i} key={i}>{item.question}</option>)}</select></label>
    <ol className="rag-steps"><li><strong>Retrieve and rank</strong><p>Illustrative matching score: shared query terms in each document’s teaching labels.</p><div className="rag-docs">{ranked.map((doc) => <label key={doc.id} className="rag-doc"><input type="checkbox" checked={selected.includes(doc.id)} onChange={() => setSelected(selected.includes(doc.id)?selected.filter((id)=>id!==doc.id):[...selected,doc.id])}/><span><strong>{doc.id}. {doc.title}</strong><small>Score {doc.score}</small><span>{doc.text}</span></span></label>)}</div></li><li><strong>Assemble context</strong><p>{selected.length ? selected.map((id) => ragDocs.find((doc) => doc.id===id)!.text).join(' ') : 'No documents selected.'}</p></li><li><strong>Answer with a source</strong><p className="rag-answer">{answer}</p></li></ol>
    <p className="lab-takeaway"><strong>What to notice:</strong> removing the relevant passage changes what can be supported. This answer is a fixed teaching rule, not a live generator; real RAG still needs retrieval evaluation and checks for unsupported claims.</p>
  </section>;
}
