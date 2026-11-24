/** Count leaf objectives once. Mapping, independent acceptance and execution differ. */
export function summarizeCoverage(sections, chapter) {
  const leaves = sections.filter(
    (s) => s.coverageUnit && (chapter === undefined || s.chapter === chapter),
  );
  const practical = leaves.filter((s) => s.requirements.implementation);
  return {
    total: leaves.length,
    mapped: leaves.filter((s) => s.mappings.length > 0).length,
    accepted: leaves.filter((s) => s.reviewStatus === "accepted").length,
    practical: practical.length,
    executed: practical.filter(
      (s) => s.realModelExecution === "verified" && s.executionRecord,
    ).length,
  };
}
