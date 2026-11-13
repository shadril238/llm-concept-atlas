import { readdirSync, readFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { spawnSync } from 'node:child_process';

const directory = resolve('src/content/docs/concepts');
let checked = 0;
for (const file of readdirSync(directory).filter((name) => name.endsWith('.mdx'))) {
  const source = readFileSync(join(directory, file), 'utf8');
  const match = source.match(/```python\n([\s\S]*?)\n```\n\n\*\*Expected output:\*\*\n\n```text\n([\s\S]*?)\n```/);
  if (!match) continue;
  const run = spawnSync('python3', ['-c', match[1]], { encoding: 'utf8', timeout: 3000 });
  if (run.error || run.status !== 0) throw new Error(`${file}: Python example failed: ${run.error?.message ?? run.stderr}`);
  if (run.stdout.trim() !== match[2].trim()) throw new Error(`${file}: documented output differs from Python output`);
  checked++;
}
if (checked !== 76) throw new Error(`Expected 76 verified Phase 5 code examples; checked ${checked}`);
console.log(`Verified ${checked} runnable Python examples against their documented output.`);
