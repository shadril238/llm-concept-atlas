import type { APIRoute } from 'astro';
import { getAtlasGraph } from '../lib/atlas';

export const GET: APIRoute = async () => new Response(JSON.stringify(await getAtlasGraph()), {
  headers: { 'Content-Type': 'application/json; charset=utf-8' },
});
