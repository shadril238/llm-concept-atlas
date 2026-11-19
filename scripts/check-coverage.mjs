import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import {
  loadCoverage,
  renderCoverageReports,
  validateCoverage,
} from "./lib/coverage.mjs";

const root = process.cwd();
const data = loadCoverage(root);
const errors = validateCoverage(data, root);
if (errors.length)
  throw new Error(
    `Coverage audit validation failed:\n- ${errors.join("\n- ")}`,
  );
const write = process.argv.includes("--write");
for (const [path, content] of Object.entries(renderCoverageReports(data))) {
  const absolute = resolve(root, path);
  if (write) writeFileSync(absolute, content);
  else if (
    !existsSync(absolute) ||
    readFileSync(absolute, "utf8") !== content
  ) {
    throw new Error(
      `${path} is stale. Run npm run coverage:report and review the result.`,
    );
  }
}
console.log(
  `Validated ${data.inventory.sections.length} source entries, ${data.audit.lessons.length} lesson audits, ${data.proposals.proposals.length} proposals, and 12 pinned notebook references. Reports ${write ? "generated" : "current"}.`,
);
