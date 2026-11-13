import cytoscape from 'cytoscape';
import type { AtlasGraph, AtlasNode } from '../lib/atlas';

type Edge = AtlasGraph['edges'][number];
type Position = { x: number; y: number };

const root = document.querySelector<HTMLElement>('[data-atlas-root]');
if (root) void startAtlas(root);

async function startAtlas(root: HTMLElement) {
  const get = <T extends Element>(selector: string): T => {
    const element = root.querySelector<T>(selector);
    if (!element) throw new Error(`Missing atlas control: ${selector}`);
    return element;
  };
  const base = root.dataset.base ?? '/';
  const canvas = get<HTMLElement>('[data-canvas]');
  const message = get<HTMLElement>('[data-graph-message]');
  const inspector = get<HTMLElement>('[data-inspector]');
  const copyButton = get<HTMLButtonElement>('[data-copy-link]');
  const copyStatus = get<HTMLElement>('[data-copy-status]');
  const search = get<HTMLInputElement>('[data-search]');
  const searchResults = get<HTMLUListElement>('[data-search-results]');
  const partFilter = get<HTMLSelectElement>('[data-filter-part]');
  const chapterFilter = get<HTMLSelectElement>('[data-filter-chapter]');
  const levelFilter = get<HTMLSelectElement>('[data-filter-level]');
  const prerequisiteToggle = get<HTMLInputElement>('[data-edge-prerequisite]');
  const relatedToggle = get<HTMLInputElement>('[data-edge-related]');
  const viewLabel = get<HTMLElement>('[data-view-label]');
  const viewDetail = get<HTMLElement>('[data-view-detail]');

  let graph: AtlasGraph;
  try {
    const response = await fetch(`${base}atlas-data.json`);
    if (!response.ok) throw new Error(`Map data returned ${response.status}`);
    graph = await response.json() as AtlasGraph;
  } catch {
    message.textContent = 'The graph could not load. Use the complete topic directory below to browse every lesson.';
    return;
  }

  const byId = new Map(graph.nodes.map((node) => [node.id, node]));
  const chapterNodes = graph.nodes.filter((node) => node.kind === 'chapter');
  const partNodes = graph.nodes.filter((node) => node.kind === 'part');
  const conceptNodes = graph.nodes.filter((node) => node.kind === 'concept');
  const edgeId = (edge: Edge, index: number) => `${edge.type}-${edge.source}-${edge.target}-${index}`;
  const edgeRecords = graph.edges.map((edge, index) => ({ ...edge, id: edgeId(edge, index) }));
  const graphLabels: Record<string, string> = {
    'chapter-01': '01  Introduction to LLMs',
    'chapter-02': '02  Tokens & Embeddings',
    'chapter-03': '03  Inside LLMs',
    'chapter-04': '04  Text Classification',
    'chapter-05': '05  Clustering & Topics',
    'chapter-06': '06  Prompt Engineering',
    'chapter-07': '07  Generation Tools',
    'chapter-08': '08  Semantic Search & RAG',
    'chapter-09': '09  Multimodal LLMs',
    'chapter-10': '10  Text Embedding Models',
    'chapter-11': '11  Fine-Tuning Encoders',
    'chapter-12': '12  Fine-Tuning Generators',
    rag: 'Retrieval-Augmented\nGeneration',
    'query-key-value': 'Queries, Keys\n& Values',
  };
  const css = getComputedStyle(root);
  const colors = {
    foundations: css.getPropertyValue('--atlas-teal').trim() || '#83e5d6',
    applications: css.getPropertyValue('--atlas-purple').trim() || '#bdacf9',
    training: css.getPropertyValue('--atlas-coral').trim() || '#ffc2ac',
  };

  const cy = cytoscape({
    container: canvas,
    elements: [
      ...graph.nodes.map((node) => ({
        data: {
          id: node.id,
          label: graphLabels[node.id] ?? node.title,
          kind: node.kind,
          part: node.part,
          lessonStatus: node.lessonStatus ?? '',
        },
        position: { x: 0, y: 0 },
      })),
      ...edgeRecords.map((edge) => ({ data: { id: edge.id, source: edge.source, target: edge.target, type: edge.type } })),
    ],
    layout: { name: 'preset', fit: false },
    style: [
      { selector: 'node', style: { 'label': 'data(label)', 'shape': 'round-rectangle', 'background-color': '#243a42', 'border-color': '#648188', 'border-width': 1.5, 'color': '#effaf8', 'font-family': 'ui-sans-serif, system-ui, sans-serif', 'font-size': 13, 'font-weight': 650, 'text-wrap': 'wrap', 'text-max-width': '160px', 'text-valign': 'center', 'text-halign': 'center', 'width': 190, 'height': 52, 'text-overflow-wrap': 'anywhere', 'overlay-opacity': 0 } },
      { selector: 'node[kind = "part"]', style: { 'width': 220, 'height': 62, 'font-size': 18, 'font-weight': 800, 'border-width': 2, 'background-color': '#1b353a' } },
      { selector: 'node[kind = "chapter"]', style: { 'width': 208, 'height': 68, 'font-size': 12, 'background-color': '#21313b', 'text-max-width': '180px' } },
      { selector: 'node[kind = "concept"]', style: { 'width': 190, 'height': 54, 'font-size': 13 } },
      { selector: 'node[part = "foundations"]', style: { 'border-color': colors.foundations } },
      { selector: 'node[part = "applications"]', style: { 'border-color': colors.applications } },
      { selector: 'node[part = "training"]', style: { 'border-color': colors.training } },
      { selector: 'node[lessonStatus = "primer"]', style: { 'border-style': 'dashed' } },
      { selector: 'edge', style: { 'width': 2, 'curve-style': 'bezier', 'line-color': '#568187', 'target-arrow-color': '#568187', 'target-arrow-shape': 'triangle', 'arrow-scale': 0.9, 'opacity': 0.82 } },
      { selector: 'edge[type = "prerequisite"]', style: { 'line-color': colors.applications, 'target-arrow-color': colors.applications, 'width': 2.5 } },
      { selector: 'edge[type = "related"]', style: { 'line-color': colors.training, 'target-arrow-color': colors.training, 'line-style': 'dashed', 'width': 2.2 } },
      { selector: '.is-hidden', style: { 'display': 'none' } },
      { selector: 'node.is-selected', style: { 'background-color': '#315e62', 'border-width': 4, 'color': '#ffffff', 'z-index': 20 } },
      { selector: 'node.is-neighbor', style: { 'background-color': '#2d4951', 'border-width': 2.5 } },
      { selector: '.is-dim', style: { 'opacity': 0.38 } },
      { selector: 'edge.is-emphasized', style: { 'width': 3.5, 'opacity': 1, 'z-index': 10 } },
    ],
    minZoom: 0.28,
    maxZoom: 2.4,
    wheelSensitivity: 0.18,
    boxSelectionEnabled: false,
    autoungrabify: true,
  });

  message.hidden = true;
  let selectedId: string | null = null;
  const mobileQuery = window.matchMedia('(max-width: 760px)');
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');

  function matchesFilters(node: AtlasNode) {
    if (partFilter.value !== 'all' && node.part !== partFilter.value) return false;
    if (chapterFilter.value !== 'all' && node.chapter !== Number(chapterFilter.value)) return false;
    if (levelFilter.value !== 'all' && (node.kind !== 'concept' || node.level !== levelFilter.value)) return false;
    return true;
  }

  function visibleState(): { ids: Set<string>; edges: Set<string>; positions: Map<string, Position> } {
    const ids = new Set<string>();
    const positions = new Map<string, Position>();
    const selected = selectedId ? byId.get(selectedId) : undefined;
    const isMobile = mobileQuery.matches;

    if (!selected) {
      const filteredChapters = chapterNodes.filter((node) => {
        if (partFilter.value !== 'all' && node.part !== partFilter.value) return false;
        if (chapterFilter.value !== 'all' && node.chapter !== Number(chapterFilter.value)) return false;
        if (levelFilter.value !== 'all') return conceptNodes.some((item) => item.chapter === node.chapter && matchesFilters(item));
        return true;
      });
      const visibleParts = partNodes.filter((node) => partFilter.value === 'all' || node.id === partFilter.value)
        .filter((node) => filteredChapters.some((chapter) => chapter.part === node.id));
      visibleParts.forEach((part, partIndex) => {
        const x = isMobile ? 480 : 270 + partIndex * 470;
        ids.add(part.id);
        positions.set(part.id, { x, y: isMobile ? 90 + partIndex * 115 : 80 });
        if (!isMobile || chapterFilter.value !== 'all') {
          filteredChapters.filter((chapter) => chapter.part === part.id).forEach((chapter, chapterIndex) => {
            ids.add(chapter.id);
            positions.set(chapter.id, { x, y: 220 + chapterIndex * 112 });
          });
        }
      });
    } else if (selected.kind === 'part') {
      ids.add(selected.id);
      positions.set(selected.id, { x: 450, y: 75 });
      chapterNodes.filter((node) => node.part === selected.id)
        .filter((node) => chapterFilter.value === 'all' || node.chapter === Number(chapterFilter.value))
        .forEach((chapter, index) => {
          ids.add(chapter.id);
          positions.set(chapter.id, { x: index % 2 === 0 ? 250 : 650, y: 235 + Math.floor(index / 2) * 120 });
        });
    } else if (selected.kind === 'chapter') {
      ids.add(selected.id);
      positions.set(selected.id, { x: 480, y: isMobile ? 85 : 170 });
      if (!isMobile) {
        ids.add(selected.part);
        positions.set(selected.part, { x: 480, y: 60 });
      }
      const localConcepts = conceptNodes.filter((node) => node.chapter === selected.chapter)
        .filter((node) => levelFilter.value === 'all' || node.level === levelFilter.value);
      localConcepts.forEach((node, index) => {
        ids.add(node.id);
        const columns = isMobile ? 2 : 3;
        positions.set(node.id, { x: isMobile ? 255 + (index % columns) * 450 : 205 + (index % columns) * 275, y: (isMobile ? 220 : 315) + Math.floor(index / columns) * 112 });
      });
    } else {
      const lineage: AtlasNode[] = [];
      let cursor: AtlasNode | undefined = selected;
      while (cursor && lineage.length < 8) {
        lineage.unshift(cursor);
        cursor = cursor.parent ? byId.get(cursor.parent) : undefined;
      }
      const displayedLineage = isMobile ? lineage.filter((node) => node.kind !== 'part') : lineage;
      displayedLineage.forEach((node, index) => {
        ids.add(node.id);
        positions.set(node.id, { x: 480, y: 80 + index * 108 });
      });
      const neighbors = edgeRecords.filter((edge) => edge.source === selected.id || edge.target === selected.id)
        .filter((edge) => edge.type === 'contains' || edge.type === 'prerequisite' && prerequisiteToggle.checked || edge.type === 'related' && relatedToggle.checked)
        .map((edge) => edge.source === selected.id ? edge.target : edge.source)
        .filter((id, index, all) => !ids.has(id) && all.indexOf(id) === index)
        .filter((id) => {
          const node = byId.get(id);
          return !!node && (levelFilter.value === 'all' || node.kind !== 'concept' || node.level === levelFilter.value);
        }).slice(0, isMobile ? 4 : 8);
      neighbors.forEach((id, index) => {
        const columns = isMobile ? 2 : 3;
        ids.add(id);
        positions.set(id, { x: isMobile ? 255 + (index % columns) * 450 : 205 + (index % columns) * 275, y: 115 + displayedLineage.length * 108 + Math.floor(index / columns) * 112 });
      });
    }

    const edges = new Set(edgeRecords.filter((edge) => {
      if (!ids.has(edge.source) || !ids.has(edge.target)) return false;
      if (edge.type === 'contains') return true;
      if (!selected || selected.kind !== 'concept') return false;
      if (edge.source !== selected.id && edge.target !== selected.id) return false;
      return edge.type === 'prerequisite' ? prerequisiteToggle.checked : relatedToggle.checked;
    }).map((edge) => edge.id));
    return { ids, edges, positions };
  }

  function fitVisible() {
    const visible = cy.elements().not('.is-hidden');
    if (visible.length) cy.fit(visible, mobileQuery.matches ? 28 : 55);
  }

  function renderMap() {
    const { ids, edges, positions } = visibleState();
    const selected = selectedId ? byId.get(selectedId) : undefined;
    message.hidden = ids.size > 0;
    if (!ids.size) message.textContent = 'No topics match these filters yet. Try another part, chapter, or difficulty level.';
    const neighborIds = new Set(edgeRecords.filter((edge) => edge.source === selectedId || edge.target === selectedId)
      .map((edge) => edge.source === selectedId ? edge.target : edge.source));
    cy.batch(() => {
      cy.nodes().forEach((element) => {
        const id = element.id();
        element.toggleClass('is-hidden', !ids.has(id));
        element.toggleClass('is-selected', id === selectedId);
        element.toggleClass('is-neighbor', id !== selectedId && neighborIds.has(id));
        element.toggleClass('is-dim', !!selected && selected.kind === 'concept' && id !== selectedId && !neighborIds.has(id) && id !== selected.parent && id !== selected.part);
        const position = positions.get(id);
        if (position) element.position(position);
      });
      cy.edges().forEach((element) => {
        const isVisible = edges.has(element.id());
        element.toggleClass('is-hidden', !isVisible);
        element.toggleClass('is-emphasized', isVisible && (element.data('source') === selectedId || element.data('target') === selectedId));
        element.toggleClass('is-dim', isVisible && !!selected && selected.kind === 'concept' && element.data('source') !== selectedId && element.data('target') !== selectedId);
      });
    });
    cy.resize();
    fitVisible();
    if (selected?.kind === 'concept') {
      viewLabel.textContent = 'Concept focus';
      viewDetail.textContent = `${selected.title} · closest links shown below`;
    } else if (selected?.kind === 'chapter') {
      viewLabel.textContent = 'Chapter focus';
      viewDetail.textContent = `Chapter ${selected.chapter} · ${conceptNodes.filter((node) => node.chapter === selected.chapter).length} concept pages`;
    } else if (selected?.kind === 'part') {
      viewLabel.textContent = 'Part focus';
      viewDetail.textContent = selected.title;
    } else {
      viewLabel.textContent = chapterFilter.value !== 'all' ? 'Filtered chapter' : mobileQuery.matches ? 'Part overview' : 'Chapter overview';
      viewDetail.textContent = chapterFilter.value !== 'all'
        ? `Chapter ${chapterFilter.value} · select its node to reveal concepts`
        : mobileQuery.matches ? 'Choose a part, then a chapter' : 'Choose a chapter to reveal its concepts';
    }
    root.querySelectorAll<HTMLButtonElement>('[data-focus-id]').forEach((button) => {
      if (button.dataset.focusId === selectedId) button.setAttribute('aria-current', 'true');
      else button.removeAttribute('aria-current');
    });
  }

  function relationGroup(label: string, ids: string[]) {
    const unique = [...new Set(ids)].filter((id) => byId.has(id));
    if (!unique.length) return null;
    const section = document.createElement('section');
    section.className = 'atlas-relation-group';
    const heading = document.createElement('h4');
    heading.textContent = label;
    const list = document.createElement('div');
    list.className = 'atlas-relation-links';
    unique.forEach((id) => {
      const node = byId.get(id)!;
      const button = document.createElement('button');
      button.type = 'button';
      button.dataset.focusId = id;
      button.textContent = node.title;
      list.append(button);
    });
    section.append(heading, list);
    return section;
  }

  function renderInspector() {
    const node = selectedId ? byId.get(selectedId) : undefined;
    copyStatus.textContent = '';
    copyButton.hidden = !node;
    if (!node) {
      const empty = document.createElement('div');
      empty.className = 'atlas-inspector-empty';
      const symbol = document.createElement('span');
      symbol.setAttribute('aria-hidden', 'true');
      symbol.textContent = '◎';
      const heading = document.createElement('h3');
      heading.textContent = 'Select a node';
      const note = document.createElement('p');
      note.textContent = 'Open a chapter or choose a concept to see its definition and relationships.';
      empty.append(symbol, heading, note);
      inspector.replaceChildren(empty);
      return;
    }
    const card = document.createElement('div');
    card.className = `atlas-inspector-card part-${node.part}`;
    const kicker = document.createElement('p');
    kicker.className = 'atlas-inspector-kicker';
    kicker.textContent = node.kind === 'part' ? 'Book part' : node.kind === 'chapter' ? `Chapter ${node.chapter}` : `Chapter ${node.chapter} · ${node.lessonStatus === 'full' ? 'Full lesson' : 'Introductory note'}`;
    const heading = document.createElement('h3');
    heading.textContent = node.title;
    const summary = document.createElement('p');
    summary.className = 'atlas-inspector-summary';
    summary.textContent = node.summary;
    const link = document.createElement('a');
    link.className = 'atlas-open-link';
    link.href = `${base}${node.href.replace(/^\/+/, '')}`;
    link.textContent = node.kind === 'concept' ? node.lessonStatus === 'full' ? 'Read full lesson ↗' : 'Read introductory note ↗' : 'Open overview ↗';
    card.append(kicker, heading, summary, link);
    const children = graph.nodes.filter((item) => item.parent === node.id).map((item) => item.id);
    const next = edgeRecords.filter((edge) => edge.type === 'prerequisite' && edge.source === node.id).map((edge) => edge.target);
    const related = edgeRecords.filter((edge) => edge.type === 'related' && (edge.source === node.id || edge.target === node.id))
      .map((edge) => edge.source === node.id ? edge.target : edge.source)
      .filter((id, index, all) => id !== node.parent && !children.includes(id) && !node.prerequisites.includes(id) && !next.includes(id) && all.indexOf(id) === index);
    const groups = [
      node.parent ? relationGroup('Inside', [node.parent]) : null,
      relationGroup('Contains', children),
      relationGroup('Learn before', node.prerequisites),
      relationGroup('Builds toward', next),
      relationGroup('Related ideas', related),
    ].filter((group): group is HTMLElement => !!group);
    if (groups.length) {
      const wrapper = document.createElement('div');
      wrapper.className = 'atlas-inspector-relations';
      wrapper.append(...groups);
      card.append(wrapper);
    }
    inspector.replaceChildren(card);
  }

  function select(id: string | null, updateUrl = true) {
    selectedId = id && byId.has(id) ? id : null;
    if (selectedId) {
      const node = byId.get(selectedId)!;
      if (partFilter.value !== 'all' && node.part !== partFilter.value) partFilter.value = 'all';
      if (chapterFilter.value !== 'all' && node.chapter !== Number(chapterFilter.value)) chapterFilter.value = 'all';
      if (levelFilter.value !== 'all' && (node.kind !== 'concept' || node.level !== levelFilter.value)) levelFilter.value = 'all';
      updateChapterOptions();
    }
    if (updateUrl) {
      const url = new URL(window.location.href);
      if (selectedId) url.searchParams.set('focus', selectedId);
      else url.searchParams.delete('focus');
      history.pushState({}, '', url);
    }
    renderMap();
    renderInspector();
  }

  function updateChapterOptions() {
    [...chapterFilter.options].forEach((option) => {
      if (option.value === 'all') return;
      option.hidden = partFilter.value !== 'all' && option.dataset.part !== partFilter.value;
      option.disabled = option.hidden;
    });
    const current = chapterFilter.selectedOptions[0];
    if (current?.disabled) chapterFilter.value = 'all';
  }

  function renderSearch() {
    const query = search.value.trim().toLocaleLowerCase();
    searchResults.replaceChildren();
    if (!query) {
      searchResults.hidden = true;
      return;
    }
    const matches = graph.nodes.filter(matchesFilters).filter((node) =>
      `${node.title} ${node.summary} ${(node.tags ?? []).join(' ')} ${node.chapter ? `chapter ${node.chapter}` : ''}`.toLocaleLowerCase().includes(query),
    ).slice(0, 8);
    if (!matches.length) {
      const item = document.createElement('li');
      item.className = 'atlas-search-empty';
      item.textContent = 'No matching topics in these filters.';
      searchResults.append(item);
    } else {
      matches.forEach((node) => {
        const item = document.createElement('li');
        const button = document.createElement('button');
        button.type = 'button';
        button.dataset.focusId = node.id;
        const title = document.createElement('strong');
        title.textContent = node.title;
        const detail = document.createElement('small');
        detail.textContent = node.kind === 'part' ? 'Book part' : node.kind === 'chapter' ? `Chapter ${node.chapter}` : `Concept · Chapter ${node.chapter}`;
        button.append(title, detail);
        item.append(button);
        searchResults.append(item);
      });
    }
    searchResults.hidden = false;
  }

  cy.on('tap', 'node', (event) => select(event.target.id()));
  cy.on('tap', (event) => { if (event.target === cy) select(null); });
  root.addEventListener('click', (event) => {
    const target = event.target as Element;
    const button = target.closest<HTMLButtonElement>('[data-focus-id]');
    if (!button || !root.contains(button)) return;
    select(button.dataset.focusId ?? null);
    searchResults.hidden = true;
    if (button.closest('.atlas-directory')) {
      get<HTMLElement>('.atlas-map-panel').scrollIntoView({ behavior: reducedMotion.matches ? 'instant' : 'smooth', block: 'start' });
    }
  });
  get<HTMLButtonElement>('[data-overview]').addEventListener('click', () => select(null));
  get<HTMLButtonElement>('[data-fit]').addEventListener('click', fitVisible);
  get<HTMLButtonElement>('[data-zoom-in]').addEventListener('click', () => cy.zoom({ level: Math.min(cy.maxZoom(), cy.zoom() * 1.25), renderedPosition: { x: canvas.clientWidth / 2, y: canvas.clientHeight / 2 } }));
  get<HTMLButtonElement>('[data-zoom-out]').addEventListener('click', () => cy.zoom({ level: Math.max(cy.minZoom(), cy.zoom() / 1.25), renderedPosition: { x: canvas.clientWidth / 2, y: canvas.clientHeight / 2 } }));
  search.addEventListener('input', renderSearch);
  search.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') { search.value = ''; renderSearch(); }
    if (event.key === 'Enter') {
      const first = searchResults.querySelector<HTMLButtonElement>('[data-focus-id]');
      if (first) { event.preventDefault(); select(first.dataset.focusId ?? null); searchResults.hidden = true; }
    }
  });
  [partFilter, chapterFilter, levelFilter].forEach((control) => control.addEventListener('change', () => {
    updateChapterOptions();
    if (selectedId && !matchesFilters(byId.get(selectedId)!)) select(null);
    else renderMap();
    renderSearch();
  }));
  [prerequisiteToggle, relatedToggle].forEach((control) => control.addEventListener('change', renderMap));
  copyButton.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(window.location.href);
      copyStatus.textContent = 'Link copied.';
    } catch {
      copyStatus.textContent = 'Copy the page address from your browser to share this topic.';
    }
  });
  window.addEventListener('popstate', () => select(new URLSearchParams(window.location.search).get('focus'), false));
  mobileQuery.addEventListener('change', renderMap);
  const resize = new ResizeObserver(() => { cy.resize(); fitVisible(); });
  resize.observe(canvas);
  updateChapterOptions();
  select(new URLSearchParams(window.location.search).get('focus'), false);
}
