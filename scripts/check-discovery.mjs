import fs from'node:fs';import assert from'node:assert/strict';import{rankRecords,prerequisiteRoute}from'../src/lib/discovery.js';
const read=p=>JSON.parse(fs.readFileSync(p,'utf8'));
const records=read('dist/discovery.json'),graph=read('dist/atlas-data.json'),cases=read('src/data/search-queries.json'),routes=read('src/data/learning-routes.json');
for(const c of cases){const results=rankRecords(records,c.query);for(const id of c.expected)assert.ok(results.slice(0,3).some(r=>r.id===id),`${c.query}: missing ${id} in top3 (${results.slice(0,3).map(r=>r.id)})`);assert.ok(!c.excludeFirst?.includes(results[0]?.id));}
for(const n of graph.nodes.filter(n=>n.kind==='concept')){
 for(const key of ['labHref','pathHref','projectHref']){assert.ok(!n[key]?.startsWith("/"),`${n.id}: route must be base-relative`);assert.ok(n[key],`${n.id} missing ${key}`);assert.ok(fs.existsSync('dist/'+n[key].split('#')[0]+'index.html'),n[key]);}
 for(const id of [...n.prerequisites,...n.related])assert.ok(n.relationshipReasons[id]?.length>15,`${n.id}/${id} missing explanation`);
 prerequisiteRoute(graph.nodes,n.id);
}
for(const route of routes)for(const id of route.steps)assert.ok(graph.nodes.some(n=>n.id===id),id);
assert.equal(rankRecords(records,'unfindablequuxtermxyz').length,0);
const lora=fs.readFileSync('dist/concepts/lora/index.html','utf8');assert.ok(lora.includes('data-pagefind-weight="4"'));assert.ok(lora.includes('kind:concept'));assert.ok(!lora.includes('AtlasExplorer.astro_astro_type_script'));
console.log(`Verified ${cases.length} query regressions, ${graph.nodes.filter(n=>n.kind==='concept').length} concept practice routes and ${routes.length} learning routes.`);

const legacy=read('docs/audits/legacy-topic-coverage.json');
for(const chapter of Object.values(legacy))for(const id of Object.values(chapter))assert.ok(graph.nodes.some(n=>n.id===id),`Archived topic lost its canonical destination: ${id}`);
