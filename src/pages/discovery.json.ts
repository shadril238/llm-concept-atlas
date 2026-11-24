import { discoveryRecords } from '../lib/learning-catalog';
export async function GET(){return new Response(JSON.stringify(await discoveryRecords()),{headers:{'Content-Type':'application/json'}});}
