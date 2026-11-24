export const progressKey = "llm-concept-atlas-progress-v2";
export const legacyKey = "llm-concept-atlas-progress-v1";
export function sanitizeProgress(value, ids, routeIds) {
  const valid = new Set(ids),
    routes = new Set(routeIds),
    object =
      value && typeof value === "object" && !Array.isArray(value) ? value : {};
  const lessons = [
    ...new Set(
      (Array.isArray(object.lessons) ? object.lessons : []).filter(
        (id) => typeof id === "string" && valid.has(id),
      ),
    ),
  ];
  /** @type {Record<string,{lab:boolean,project:boolean}>} */
  const practice = {};
  for (const [id, v] of Object.entries(object.practice ?? {}))
    if (routes.has(id) && v && typeof v === "object")
      practice[id] = { lab: v.lab === true, project: v.project === true };
  return {
    version: 2,
    lessons,
    practice,
    lastConceptId: valid.has(object.lastConceptId)
      ? object.lastConceptId
      : null,
  };
}
export function loadProgress(storage, ids, routeIds) {
  const parse = (s) => {
    try {
      return JSON.parse(s ?? "null");
    } catch {
      return null;
    }
  };
  try {
    const current = parse(storage.getItem(progressKey));
    if (current?.version && current.version !== 2)
      return {
        data: sanitizeProgress(current, ids, routeIds),
        persistent: false,
        migrated: false,
        protectedStorage: true,
      };
    if (current?.version === 2)
      return {
        data: sanitizeProgress(current, ids, routeIds),
        persistent: true,
        migrated: false,
      };
    const legacy = parse(storage.getItem(legacyKey));
    const data = sanitizeProgress(
      { lessons: Array.isArray(legacy) ? legacy : [] },
      ids,
      routeIds,
    );
    // Retain the old key as a backup. Unknown IDs never inflate current completion.
    storage.setItem(progressKey, JSON.stringify(data));
    return { data, persistent: true, migrated: Array.isArray(legacy) };
  } catch {
    return {
      data: sanitizeProgress({}, ids, routeIds),
      persistent: false,
      migrated: false,
    };
  }
}
export function reduceProgress(data, event, ids, routeIds) {
  const next = sanitizeProgress(data, ids, routeIds);
  if (event.type === "lesson" && ids.includes(event.id)) {
    const set = new Set(next.lessons);
    set.has(event.id) ? set.delete(event.id) : set.add(event.id);
    next.lessons = [...set];
  }
  if (event.type === "visit" && ids.includes(event.id))
    next.lastConceptId = event.id;
  if (
    event.type === "practice" &&
    routeIds.includes(event.id) &&
    ["lab", "project"].includes(event.step)
  )
    next.practice[event.id] = {
      ...next.practice[event.id],
      [event.step]: event.checked === true,
    };
  return sanitizeProgress(next, ids, routeIds);
}
