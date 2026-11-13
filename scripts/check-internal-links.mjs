import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join, relative, resolve } from 'node:path';

const dist = resolve('dist');
const origin = 'https://shadril238.github.io';
const base = '/llm-concept-atlas/';
const problems = [];
let checked = 0;

function htmlFiles(directory) {
  return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const path = join(directory, entry.name);
    return entry.isDirectory() ? htmlFiles(path) : entry.name.endsWith('.html') ? [path] : [];
  });
}

for (const file of htmlFiles(dist)) {
  const html = readFileSync(file, 'utf8');
  const pagePath = relative(dist, file).replace(/index\.html$/, '');
  const pageUrl = new URL(base + pagePath, origin);
  for (const match of html.matchAll(/\b(?:href|src)=(?:"([^"]*)"|'([^']*)')/g)) {
    const raw = (match[1] ?? match[2]).replaceAll('&amp;', '&');
    if (!raw || raw.startsWith('data:') || raw.startsWith('mailto:') || raw.startsWith('tel:')) continue;
    let url;
    try { url = new URL(raw, pageUrl); } catch { problems.push(`${pagePath}: invalid URL ${raw}`); continue; }
    if (url.origin !== origin) continue;
    // Starlight gives the generated GitHub Pages 404.html a virtual /404/ canonical URL.
    if (pagePath === '404.html' && url.pathname === `${base}404/`) continue;
    checked++;
    if (!url.pathname.startsWith(base)) {
      problems.push(`${pagePath}: missing repository base in ${raw}`);
      continue;
    }
    let targetPath;
    try { targetPath = decodeURIComponent(url.pathname.slice(base.length)); }
    catch { problems.push(`${pagePath}: invalid encoding in ${raw}`); continue; }
    const target = resolve(dist, targetPath, targetPath.endsWith('/') || !targetPath.includes('.') ? 'index.html' : '');
    if (!target.startsWith(dist + '/') || !statFile(target)) {
      problems.push(`${pagePath}: missing target ${raw}`);
      continue;
    }
    if (url.hash && target.endsWith('.html')) {
      const id = decodeURIComponent(url.hash.slice(1));
      const targetHtml = target === file ? html : readFileSync(target, 'utf8');
      if (!targetHtml.includes(`id="${id}"`) && !targetHtml.includes(`id='${id}'`)) {
        problems.push(`${pagePath}: missing anchor ${raw}`);
      }
    }
  }
}

function statFile(path) {
  try { return statSync(path).isFile(); } catch { return false; }
}

if (problems.length) {
  console.error(`Found ${problems.length} broken internal references:\n${problems.slice(0, 30).join('\n')}`);
  process.exitCode = 1;
} else {
  console.log(`Checked ${checked} internal links and assets across the built HTML pages.`);
}
