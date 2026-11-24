import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, readdirSync, existsSync } from 'node:fs';
import { join, resolve } from 'node:path';

const docs = resolve('src/content/docs');
const concepts = join(docs, 'concepts');
const files = readdirSync(concepts).filter((name) => name.endsWith('.mdx'));
const inventory = JSON.parse(readFileSync(resolve('src/data/book-sections.json'), 'utf8'));

test('published concepts retain the existing lesson structure (not a depth certification)', () => {
  assert.ok(files.length > 0);
  for (const file of files) {
    const content = readFileSync(join(concepts, file), 'utf8');
    const [, frontmatter, body] = content.match(/^---\n([\s\S]*?)\n---\n([\s\S]*)$/) ?? [];
    assert.ok(body, `${file} needs valid frontmatter`);
    assert.match(frontmatter, /lessonStatus: full/, `${file} needs full status`);
    assert.ok((body.match(/^## /gm) ?? []).length >= 7, `${file} needs a teaching sequence`);
    assert.ok((body.match(/<details>/g) ?? []).length >= 2, `${file} needs self-checks`);
    assert.match(body, /```python\n/, `${file} needs runnable code`);
    assert.ok((body.match(/\b\w+\b/g) ?? []).length >= 350, `${file} needs depth`);
  }
});

test('chapter navigation uses audited source objectives, not the retired topic map', () => {
  for(let chapter=1;chapter<=12;chapter++){
    const leaves=inventory.sections.filter(s=>s.chapter===chapter&&s.coverageUnit);
    assert.ok(leaves.length>0);
    for(const leaf of leaves)for(const mapping of leaf.mappings)assert.ok(existsSync(resolve(mapping.path)),mapping.path);
  }
  assert.ok(!readFileSync(resolve('src/components/ChapterOverview.astro'),'utf8').includes('topic-coverage'));
});

test('internal lesson and path links resolve to source pages', () => {
  function visit(directory) {
    return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
      const path = join(directory, entry.name);
      return entry.isDirectory() ? visit(path) : entry.name.endsWith('.mdx') ? [path] : [];
    });
  }
  for (const file of visit(docs)) {
    const content = readFileSync(file, 'utf8');
    for (const [, target] of content.matchAll(/\]\((\/llm-concept-atlas\/[^)#]+)(?:#[^)]*)?\)/g)) {
      const slug = target.replace(/^\/llm-concept-atlas\//, '').replace(/\/$/, '');
      const found = existsSync(join(docs, `${slug}.mdx`)) ||
        existsSync(join(docs, slug, 'index.mdx')) ||
        slug === 'atlas';
      assert.ok(found, `${file}: ${target} has no page`);
    }
  }
});
