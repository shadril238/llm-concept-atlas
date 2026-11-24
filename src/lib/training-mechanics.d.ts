export function crossEntropy(logits: number[], target: number): number;
export function multipleNegatives(
  scores: number[][],
  scale?: number,
): { terms: number[]; loss: number };
export function pairFeatures(a: number[], b: number[]): number[];
export function cosineRegression(
  scores: number[],
  targets: number[],
): { terms: number[]; loss: number };
export function supervisedMask(
  tokens: { original_id: number; special_or_padding: boolean }[],
  selected: number[],
): number[];
