import { useState } from 'react';

function LabFrame({ title, label, children, onReset }: { title: string; label: string; children: React.ReactNode; onReset: () => void }) {
  return <section className="learning-lab phase-five-lab" aria-label={title}>
    <div className="lab-heading"><div><span className="eyebrow">Interactive lab · {label}</span><h3>{title}</h3></div><button type="button" onClick={onReset}>Reset</button></div>
    {children}
  </section>;
}

function Bar({ value, label }: { value: number; label: string }) {
  return <div className="lab-bar" aria-label={`${label}: ${Math.round(value * 100)} percent`}><span style={{ width: `${Math.max(0, Math.min(100, value * 100))}%` }} /></div>;
}

const reviews = [
  { text: 'The camera is excellent, but the battery is weak.', positive: 0.48, truth: 'mixed' },
  { text: 'A fast, reliable laptop that I enjoy using.', positive: 0.86, truth: 'positive' },
  { text: 'The app crashes whenever I save my work.', positive: 0.12, truth: 'negative' },
];

export function ClassificationLab() {
  const [review, setReview] = useState(0);
  const [threshold, setThreshold] = useState(0.5);
  const item = reviews[review];
  const prediction = item.positive >= threshold ? 'positive' : 'negative';
  return <LabFrame title="Where does the label change?" label="fixed example scores" onReset={() => { setReview(0); setThreshold(0.5); }}>
    <p>Select a review and move the threshold. These scores are invented to expose the decision rule; no trained classifier runs in this browser.</p>
    <div className="lab-controls"><label>Review <select value={review} onChange={(event) => setReview(Number(event.target.value))}>{reviews.map((item, index) => <option value={index} key={index}>{item.text.slice(0, 38)}…</option>)}</select></label><label>Positive threshold <input type="range" min="0.1" max="0.9" step="0.05" value={threshold} onChange={(event) => setThreshold(Number(event.target.value))} /> {threshold.toFixed(2)}</label></div>
    <blockquote className="lab-sample">“{item.text}”</blockquote>
    <div className="lab-metric"><span>Illustrative positive score</span><Bar value={item.positive} label="Positive score" /><strong>{item.positive.toFixed(2)}</strong></div>
    <p className="lab-result" aria-live="polite">Predicted label: <strong>{prediction}</strong>. Human reading: <strong>{item.truth}</strong>.</p>
    <p className="lab-takeaway"><strong>What to notice:</strong> moving a threshold changes the decision without changing the text or model score. The mixed review exposes a limitation of forcing two labels onto a nuanced input.</p>
  </LabFrame>;
}

const clusterPoints = [
  { name: 'Rain forecast', x: 16, y: 24 }, { name: 'Storm warning', x: 25, y: 32 },
  { name: 'Cloud cover', x: 34, y: 22 }, { name: 'Beach guide', x: 76, y: 22 },
  { name: 'Sunny vacation', x: 82, y: 34 }, { name: 'Weather travel', x: 52, y: 44 },
];
const clusterCenters = [{ x: 24, y: 26 }, { x: 79, y: 28 }, { x: 51, y: 46 }];
const clusterNames = ['Weather', 'Travel', 'Bridge'];

export function ClusteringLab() {
  const [count, setCount] = useState(2);
  const [position, setPosition] = useState(52);
  const points = clusterPoints.map((point, index) => index === 5 ? { ...point, x: position } : point);
  const assignments = points.map((point) => clusterCenters.slice(0, count).map((center) => Math.hypot(point.x - center.x, point.y - center.y)).indexOf(Math.min(...clusterCenters.slice(0, count).map((center) => Math.hypot(point.x - center.x, point.y - center.y)))));
  return <LabFrame title="Move a document between clusters" label="invented 2D document vectors" onReset={() => { setCount(2); setPosition(52); }}>
    <p>Each point stands for a document embedding. The fixed centers are teaching choices; the nearest center assigns the color. Real clustering learns structure from high-dimensional data and may mark noise.</p>
    <div className="lab-controls"><label>Number of centers <select value={count} onChange={(event) => setCount(Number(event.target.value))}><option value={2}>2</option><option value={3}>3</option></select></label><label>Bridge document position <input type="range" min="30" max="75" value={position} onChange={(event) => setPosition(Number(event.target.value))} /> {position}</label></div>
    <div className="lab-grid"><svg className="cluster-svg" viewBox="0 0 440 250" role="img" aria-label="Scatter plot of six invented document vectors and fixed cluster centers">
      <line x1="35" y1="215" x2="420" y2="215" className="axis" /><line x1="35" y1="15" x2="35" y2="215" className="axis" />
      {clusterCenters.slice(0, count).map((center, index) => <g key={index}><rect x={35 + center.x * 3.7 - 5} y={215 - center.y * 3.7 - 5} width="10" height="10" className={`cluster-center group-${index}`} /><text x={35 + center.x * 3.7 + 8} y={215 - center.y * 3.7 + 4} className="svg-label">{clusterNames[index]}</text></g>)}
      {points.map((point, index) => <circle key={point.name} cx={35 + point.x * 3.7} cy={215 - point.y * 3.7} r="7" className={`cluster-point group-${assignments[index]}`}><title>{`${point.name}: ${clusterNames[assignments[index]]}`}</title></circle>)}
    </svg><div><h4>Assigned groups</h4><ul className="lab-assignments">{points.map((point, index) => <li key={point.name}><span>{point.name}</span><strong>{clusterNames[assignments[index]]}</strong></li>)}</ul></div></div>
    <p className="lab-takeaway"><strong>What to notice:</strong> the borderline document can switch groups after a tiny move or when a third center is introduced. Read actual documents before naming a cluster as a topic.</p>
  </LabFrame>;
}

export function PromptLab() {
  const [examples, setExamples] = useState(true);
  const [format, setFormat] = useState<'plain' | 'json'>('plain');
  const [query, setQuery] = useState('The delivery was late.');
  const instruction = format === 'json' ? 'Return JSON with exactly one field named label: positive or negative.' : 'Return exactly one word: positive or negative.';
  const prompt = [instruction, 'Classify the sentiment of the customer message.', examples && 'Example: The package arrived early. → positive\nExample: The parcel was damaged. → negative', `Message: ${query}`, format === 'json' ? 'JSON:' : 'Label:'].filter(Boolean).join('\n\n');
  return <LabFrame title="Assemble a prompt" label="prompt construction, no model call" onReset={() => { setExamples(true); setFormat('plain'); setQuery('The delivery was late.'); }}>
    <p>Switch the demonstration examples and output contract. The preview shows exactly what a model would receive; it does not pretend to generate an answer.</p>
    <div className="lab-controls"><label><input type="checkbox" checked={examples} onChange={(event) => setExamples(event.target.checked)} /> Include examples</label><label>Output <select value={format} onChange={(event) => setFormat(event.target.value as 'plain' | 'json')}><option value="plain">One label</option><option value="json">JSON object</option></select></label></div>
    <label className="lab-text-input">New message <input value={query} onChange={(event) => setQuery(event.target.value)} maxLength={120} /></label>
    <pre className="lab-prompt-preview" aria-live="polite">{prompt}</pre>
    <p className="lab-takeaway"><strong>What to notice:</strong> examples clarify the mapping but spend context tokens. A JSON instruction specifies shape; a real application still validates the parsed response and checks the label.</p>
  </LabFrame>;
}

const scenes = [
  { name: 'Dog on grass', vector: [0.85, 0.18], captions: ['a dog outdoors', 'a car on a road', 'a cat indoors'] },
  { name: 'Car on road', vector: [-0.7, 0.5], captions: ['a car on a road', 'a dog outdoors', 'a red bicycle'] },
];
const captionVectors: Record<string, [number, number]> = {
  'a dog outdoors': [0.82, 0.25], 'a car on a road': [-0.72, 0.49],
  'a cat indoors': [0.62, -0.3], 'a red bicycle': [-0.55, 0.32],
};
const cosine = (a: number[], b: number[]) => a.reduce((sum, value, index) => sum + value * b[index], 0) / (Math.hypot(...a) * Math.hypot(...b));

export function MultimodalLab() {
  const [sceneIndex, setSceneIndex] = useState(0);
  const scene = scenes[sceneIndex];
  const scores = scene.captions.map((caption) => ({ caption, score: cosine(scene.vector, captionVectors[caption]) })).sort((a, b) => b.score - a.score);
  return <LabFrame title="Match an image to text" label="invented shared vectors" onReset={() => setSceneIndex(0)}>
    <p>Choose a described scene. Image and caption vectors share an invented 2D space; cosine ranks the captions. This is matching, not image caption generation.</p>
    <div className="lab-controls"><label>Illustrated scene <select value={sceneIndex} onChange={(event) => setSceneIndex(Number(event.target.value))}>{scenes.map((item, index) => <option key={item.name} value={index}>{item.name}</option>)}</select></label></div>
    <div className="lab-grid"><div className={`scene-card scene-${sceneIndex}`} role="img" aria-label={`Simple symbolic illustration: ${scene.name}`}><span aria-hidden="true">{sceneIndex === 0 ? '🐕' : '🚗'}</span><strong>{scene.name}</strong></div><div><h4>Caption similarity</h4><ul className="lab-scores">{scores.map((item) => <li key={item.caption}><span>{item.caption}</span><Bar value={(item.score + 1) / 2} label={`${item.caption} cosine similarity`} /><strong>{item.score.toFixed(2)}</strong></li>)}</ul></div></div>
    <p className="lab-takeaway"><strong>What to notice:</strong> changing the image vector changes the nearest caption. High similarity means alignment in this space; it does not prove that every detail in a caption is present.</p>
  </LabFrame>;
}

export function ContrastiveLab() {
  const [positive, setPositive] = useState(0.7);
  const [negative, setNegative] = useState(0.35);
  const [temperature, setTemperature] = useState(0.5);
  const probability = Math.exp(positive / temperature) / (Math.exp(positive / temperature) + Math.exp(negative / temperature));
  const loss = -Math.log(probability);
  return <LabFrame title="Pull a positive pair ahead" label="two-pair contrastive loss" onReset={() => { setPositive(0.7); setNegative(0.35); setTemperature(0.5); }}>
    <p>Adjust similarities for a question-answer positive pair and a competing negative. The loss is negative log probability of choosing the positive in this tiny batch.</p>
    <div className="lab-controls"><label>Positive similarity <input type="range" min="-1" max="1" step="0.05" value={positive} onChange={(event) => setPositive(Number(event.target.value))} /> {positive.toFixed(2)}</label><label>Negative similarity <input type="range" min="-1" max="1" step="0.05" value={negative} onChange={(event) => setNegative(Number(event.target.value))} /> {negative.toFixed(2)}</label><label>Temperature <input type="range" min="0.1" max="1" step="0.05" value={temperature} onChange={(event) => setTemperature(Number(event.target.value))} /> {temperature.toFixed(2)}</label></div>
    <div className="lab-metric"><span>Positive chosen</span><Bar value={probability} label="Probability assigned to positive pair" /><strong>{(probability * 100).toFixed(0)}%</strong></div>
    <p className="lab-result" aria-live="polite">Teaching loss: <strong>{loss.toFixed(3)}</strong>. Lower is better for this pair choice.</p>
    <p className="lab-takeaway"><strong>What to notice:</strong> increasing positive similarity or reducing negative similarity lowers loss. False negatives are harmful because the objective then pushes a true match away.</p>
  </LabFrame>;
}

export function AdapterLab() {
  const [rank, setRank] = useState(1);
  const [bits, setBits] = useState(16);
  const [size, setSize] = useState(4096);
  const full = size * size;
  const adapter = 2 * size * rank;
  const baseMiB = full * bits / 8 / 1048576;
  const adapterMiB = adapter * 2 / 1048576;
  return <LabFrame title="Compare a full matrix with LoRA" label="parameter and storage arithmetic" onReset={() => { setRank(1); setBits(16); setSize(4096); }}>
    <p>One square projection matrix is used as an example. LoRA trains two thin matrices instead of changing every base weight. Choose a rank and frozen-base precision to see the parameter counts.</p>
    <div className="lab-controls"><label>Matrix width <select value={size} onChange={(event) => setSize(Number(event.target.value))}><option value={1024}>1,024</option><option value={4096}>4,096</option><option value={8192}>8,192</option></select></label><label>LoRA rank <input type="range" min="1" max="64" step="1" value={rank} onChange={(event) => setRank(Number(event.target.value))} /> {rank}</label><label>Frozen base <select value={bits} onChange={(event) => setBits(Number(event.target.value))}><option value={16}>16-bit</option><option value={8}>8-bit</option><option value={4}>4-bit</option></select></label></div>
    <dl className="lab-stats"><div><dt>Full matrix</dt><dd>{full.toLocaleString()} weights</dd></div><div><dt>LoRA trainable</dt><dd>{adapter.toLocaleString()} weights ({(100 * adapter / full).toFixed(2)}%)</dd></div><div><dt>Frozen base storage</dt><dd>~{baseMiB.toFixed(1)} MiB</dd></div><div><dt>Adapter storage at 16-bit</dt><dd>~{adapterMiB.toFixed(2)} MiB</dd></div></dl>
    <p className="lab-takeaway"><strong>What to notice:</strong> lower base precision changes storage, while rank changes trainable adapter size. This counts one matrix only; real QLoRA also needs activations, optimizer state, quantization metadata, and compatible kernels.</p>
  </LabFrame>;
}

export function PreferenceLab() {
  const [chosen, setChosen] = useState(-0.4);
  const [rejected, setRejected] = useState(-1.1);
  const [beta, setBeta] = useState(0.5);
  const referenceGap = 0.2;
  const improvement = (chosen - rejected) - referenceGap;
  const loss = Math.log(1 + Math.exp(-beta * improvement));
  return <LabFrame title="Compare preferred responses" label="illustrative DPO objective" onReset={() => { setChosen(-0.4); setRejected(-1.1); setBeta(0.5); }}>
    <p>Imagine one prompt with a preferred cited answer and a rejected unsupported answer. Move their policy log probabilities; DPO compares their gap with a fixed reference gap of 0.2.</p>
    <div className="lab-controls"><label>Chosen log probability <input type="range" min="-2" max="0" step="0.1" value={chosen} onChange={(event) => setChosen(Number(event.target.value))} /> {chosen.toFixed(1)}</label><label>Rejected log probability <input type="range" min="-2" max="0" step="0.1" value={rejected} onChange={(event) => setRejected(Number(event.target.value))} /> {rejected.toFixed(1)}</label><label>β strength <input type="range" min="0.1" max="1" step="0.1" value={beta} onChange={(event) => setBeta(Number(event.target.value))} /> {beta.toFixed(1)}</label></div>
    <p className="lab-result" aria-live="polite">Policy gap: <strong>{(chosen - rejected).toFixed(2)}</strong>. Improvement over reference: <strong>{improvement.toFixed(2)}</strong>. Teaching DPO loss: <strong>{loss.toFixed(3)}</strong>.</p>
    <p className="lab-takeaway"><strong>What to notice:</strong> loss decreases as the chosen response gains relative likelihood beyond the reference gap. The arithmetic cannot judge whether a human preference label is good; pair quality and held-out evaluation remain essential.</p>
  </LabFrame>;
}
