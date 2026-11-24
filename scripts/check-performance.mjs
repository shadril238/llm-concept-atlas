/** File-size budgets for the production output, not simulated network/CPU scores. */
import fs from "node:fs";
import path from "node:path";
import zlib from "node:zlib";
import assert from "node:assert/strict";
export function payload(root, slug) {
  const html = fs.readFileSync(path.join(root, slug, "index.html"), "utf8");
  const seeds = [
    ...html.matchAll(
      /(?:src|href|component-url|renderer-url)="([^"]+\.js(?:\?[^"]*)?)"/g,
    ),
  ].map((m) => m[1]);
  const seen = new Set();
  function visit(url) {
    const clean = url.replace("/llm-concept-atlas/", "").replace(/\?.*$/, "");
    const p = path.join(root, clean);
    if (seen.has(p) || !fs.existsSync(p) || !p.endsWith(".js")) return;
    seen.add(p);
    const source = fs.readFileSync(p, "utf8");
    for (const m of source.matchAll(
      /(?:from\s*|import\s*\(?\s*)["'`](\.\.?\/[^"'`]+\.js)["'`]/g,
    ))
      visit(path.relative(root, path.resolve(path.dirname(p), m[1])));
  }
  seeds.forEach(visit);
  return {
    htmlBytes: Buffer.byteLength(html),
    reachableJsBytes: [...seen].reduce((n, p) => n + fs.statSync(p).size, 0),
    reachableJsGzipBytes: [...seen].reduce(
      (n, p) => n + zlib.gzipSync(fs.readFileSync(p)).length,
      0,
    ),
    modules: [...seen].map((p) => path.relative(root, p)).sort(),
  };
}
if (process.argv[1]?.endsWith("check-performance.mjs")) {
  const root = process.argv[2] ?? "dist",
    lora = payload(root, "concepts/lora"),
    graph = payload(root, "atlas"),
    discovery = fs.readFileSync(path.join(root, "discovery.json"));
  const current = {
    method:
      "HTML-linked and recursively reachable JavaScript, including lazy islands/imports; gzip file estimates, not measured network bytes or interaction timing.",
    lora,
    graph,
    discovery: {
      bytes: discovery.length,
      gzipBytes: zlib.gzipSync(discovery).length,
    },
  };
  assert.ok(
    lora.reachableJsGzipBytes < 180000,
    "Ordinary LoRA lesson exceeds 180KB gzip JS budget",
  );
  assert.ok(
    graph.reachableJsGzipBytes < 250000,
    "Graph exceeds 250KB gzip reachable JS budget",
  );
  assert.ok(
    current.discovery.gzipBytes < 45000,
    "Discovery catalog exceeds 45KB gzip budget",
  );
  assert.ok(
    !lora.modules.some((p) => /cytoscape|AtlasExplorer/i.test(p)),
    "Ordinary lesson includes graph implementation",
  );
  if (process.argv.includes("--write"))
    fs.writeFileSync(
      "docs/performance/e9-production.json",
      JSON.stringify(current, null, 2) + "\n",
    );
  console.log(
    `Payload budgets passed: LoRA ${lora.reachableJsGzipBytes}B, graph ${graph.reachableJsGzipBytes}B reachable gzip JS; catalog ${current.discovery.gzipBytes}B gzip.`,
  );
}
