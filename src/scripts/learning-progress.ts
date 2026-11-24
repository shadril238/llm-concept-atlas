import {
  loadProgress,
  progressKey,
  reduceProgress,
} from "../lib/learning-progress.js";
const config = document.querySelector<HTMLScriptElement>(
  "[data-progress-config]",
);
if (config) {
  const { ids, routes, current, base } = JSON.parse(config.textContent ?? "{}");
  const storage = {
    getItem: (key: string) => window.localStorage.getItem(key),
    setItem: (key: string, value: string) =>
      window.localStorage.setItem(key, value),
  };
  let state = loadProgress(storage, ids, routes);
  function save(event: any) {
    state.data = reduceProgress(state.data, event, ids, routes);
    try {
      if ("protectedStorage" in state && state.protectedStorage)
        throw new Error("Newer progress format preserved");
      localStorage.setItem(progressKey, JSON.stringify(state.data));
      state.persistent = true;
    } catch {
      state.persistent = false;
    }
    render();
  }
  function render() {
    for (const root of document.querySelectorAll<HTMLElement>(
      "[data-lesson-progress]",
    )) {
      const id = root.dataset.conceptId!,
        button = root.querySelector("button")!,
        count = root.querySelector<HTMLElement>(".lesson-progress-count")!,
        done = state.data.lessons.includes(id);
      button.setAttribute("aria-pressed", String(done));
      button.textContent = done ? "Marked read" : "Mark lesson as read";
      count.textContent = `${state.data.lessons.length} of ${ids.length} lessons marked read`;
    }
    for (const el of document.querySelectorAll<HTMLElement>(
      "[data-progress-summary]",
    ))
      el.textContent = `${state.data.lessons.length} of ${ids.length} lessons marked read. ${state.persistent ? "Saved in this browser." : "Changes stay on this page; existing saved data is preserved."}${state.migrated ? " Your previous reading checkmarks were preserved." : ""}`;
    for (const el of document.querySelectorAll<HTMLElement>(
      "[data-route-progress]",
    )) {
      const steps = JSON.parse(el.dataset.concepts ?? "[]");
      el.textContent = `${steps.filter((id: string) => state.data.lessons.includes(id)).length} of ${steps.length} route lessons marked read`;
    }
    for (const el of document.querySelectorAll<HTMLInputElement>(
      "[data-practice-step]",
    ))
      el.checked = Boolean(
        state.data.practice[el.dataset.routeId!]?.[
          el.dataset.practiceStep as "lab" | "project"
        ],
      );
    for (const el of document.querySelectorAll<HTMLAnchorElement>(
      "[data-resume]",
    )) {
      el.hidden = !state.data.lastConceptId;
      if (state.data.lastConceptId)
        el.href = base + "concepts/" + state.data.lastConceptId + "/";
    }
    for (const el of document.querySelectorAll<HTMLElement>(
      "[data-storage-note]",
    ))
      el.textContent = state.persistent
        ? "Progress stays in this browser. Reading and practice are self-reported."
        : "Progress cannot be saved here. Checkmarks remain available on this page, and existing saved data is preserved.";
  }
  document
    .querySelectorAll<HTMLElement>("[data-lesson-progress]")
    .forEach((root) =>
      root
        .querySelector("button")
        ?.addEventListener("click", () =>
          save({ type: "lesson", id: root.dataset.conceptId }),
        ),
    );
  document
    .querySelectorAll<HTMLInputElement>("[data-practice-step]")
    .forEach((input) =>
      input.addEventListener("change", () =>
        save({
          type: "practice",
          id: input.dataset.routeId,
          step: input.dataset.practiceStep,
          checked: input.checked,
        }),
      ),
    );
  window.addEventListener("storage", (e) => {
    if (e.key === progressKey) {
      state = loadProgress(storage, ids, routes);
      render();
    }
  });
  if (current) save({ type: "visit", id: current });
  else render();
}
