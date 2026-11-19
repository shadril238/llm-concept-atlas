import { spawnSync } from 'node:child_process';
const result = spawnSync('python3', ['scripts/run-examples.py'], { stdio: 'inherit' });
process.exit(result.status ?? 1);
