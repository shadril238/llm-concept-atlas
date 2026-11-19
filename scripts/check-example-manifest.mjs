import { spawnSync } from 'node:child_process';
const result = spawnSync('python3', ['scripts/sync-examples.py', '--check'], { stdio: 'inherit' });
process.exit(result.status ?? 1);
