export function aggregateAnswers(samples: (number | null)[]): {
  winner: number | null;
  invalid: number;
  counts: Record<string, number>;
  strict_majority: boolean;
  total: number;
};
export function validateOrder(raw: string): {
  valid: boolean;
  value: { item: string; quantity: number; priority: string } | null;
  error: string | null;
};
