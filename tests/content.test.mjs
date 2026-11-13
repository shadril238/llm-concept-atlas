import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, readdirSync, existsSync } from 'node:fs';
import { join, resolve } from 'node:path';

const docs = resolve('src/content/docs');
const concepts = join(docs, 'concepts');
const files = readdirSync(concepts).filter((name) => name.endsWith('.mdx'));
const coverage = JSON.parse(readFileSync(resolve('src/data/topic-coverage.json'), 'utf8'));

test('every published concept is a complete teaching lesson', () => {
  assert.equal(files.length, 96);
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

test('every chapter topic has a full concept lesson', () => {
  const mapSource = readFileSync(resolve('src/data/book-map.ts'), 'utf8');
  const chapterMatches = [...mapSource.matchAll(/number: (\d+),[\s\S]*?topics: \[([\s\S]*?)\]/g)];
  assert.equal(chapterMatches.length, 12);
  for (const [, chapter, body] of chapterMatches) {
    const listedTopics = [...body.matchAll(/"([^"]+)"/g)].map((match) => match[1]);
    assert.deepEqual(Object.keys(coverage[chapter]), listedTopics, `chapter ${chapter} coverage drifted from the book map`);
  }
  const mapped = Object.values(coverage).flatMap((chapter) => Object.values(chapter));
  assert.equal(mapped.length, 95);
  for (const conceptId of mapped) {
    const file = join(concepts, `${conceptId}.mdx`);
    assert.ok(existsSync(file), `${conceptId} has no lesson`);
    assert.match(readFileSync(file, 'utf8'), /lessonStatus: full/, `${conceptId} is not a full lesson`);
  }
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
