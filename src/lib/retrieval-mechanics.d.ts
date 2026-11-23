export function rankingMetrics(
  ranking: string[],
  relevance: Record<string, number>,
  k?: number,
): {
  precision: number;
  recall: number | null;
  ap: number | null;
  rr: number | null;
  ndcg: number | null;
  hits: number;
  relevant_documents: number;
  ap_terms: number[];
  dcg: number;
  idcg: number;
};
export function makeChunks(
  text: string,
  size?: number,
  overlap?: number,
  mode?: string,
): { start: number; end: number; text: string }[];
export function candidateSoftmax(scores: number[], scale?: number): number[];
export function contrastiveLoss(logits: number[][]): {
  image_to_text: number;
  text_to_image: number;
  loss: number;
  image_terms: number[];
  text_terms: number[];
};
export function citationMetrics(
  claims: {
    fully_supported_by_citations: boolean;
    citations: { supports: boolean }[];
  }[],
): {
  citation_precision: number | null;
  citation_recall: number | null;
  claims: number;
  citations: number;
};
