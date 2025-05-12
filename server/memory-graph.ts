import * as faiss from "faiss-node";
import { logger } from "./logger";
import { inspect } from "util";
import * as yaml from 'js-yaml';
import { v4 as uuid } from "uuid";

/* ───────── Your existing domain types ───────── */
export type EntityRelation = [string, string];

export interface Entity {
  id: string;
  type: string;
  name: string;
  appearance?: string;
  clothes?: string;
  info?: string;
  secret?: string;
  state?: string;
}

export interface MemoryChunk {
  id: string;
  text: string;
  meta: { type: string };
}

const entityToChunk = (e: Entity): MemoryChunk => ({
  id: e.id,
  text: yaml.dump(e),
  meta: { type: "entity" }
});

function narrativeToChunks(turnNumber: number, narrative: string): MemoryChunk[] {
  const paragraphs = narrative.split(/\n\s*\n/).map(chunk => chunk.trim().replace(/\n/g, " "));
  const pairedChunks: string[] = [];
  
  for (let i = 0; i < paragraphs.length; i += 2) {
    if (i + 1 < paragraphs.length) {
      pairedChunks.push(`${paragraphs[i]} ${paragraphs[i + 1]}`);
    } else {
      pairedChunks.push(paragraphs[i]);
    }
  }
  
  return pairedChunks.map((chunk, index) => ({
    id: `narrative-${turnNumber}-${index}`,
    text: `Turn ${turnNumber} p${index + 1}: ${chunk}`,
    meta: { type: "narrative" }
  }));
}

export function extractTurnNumber(chunkId: string): number | undefined {
  const match = chunkId.match(/narrative-(\d+)-\d+/);
  if (!match) return undefined;
  return parseInt(match[1], 10);
}


/* ───────── Vector store ───────── */
export class MemoryVectorStore {
  private dim = 384;
  private index = new faiss.IndexFlatL2(this.dim);

  /** Mean‑pooled + L2‑norm embedder will be set in `init()` */
  private embed!: (txt: string) => Promise<Float32Array>;

  private rowToChunk = new Map<number, MemoryChunk>(); // FAISS row → chunk

  async init() {
    const { pipeline } = await import("@xenova/transformers"); // dynamic import → CJS safe
    const raw = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");

    // tiny wrapper that returns a *single* 384‑float vector
    this.embed = async (txt: string) => {
      const t: any = await raw(txt, { pooling: "mean", normalize: true });
      return t.data as Float32Array; // shape [384]
    };
  }

  async upsertChunk(chunk: MemoryChunk) {
    const vec = await this.embed(chunk.text);       // Float32Array(384)

    try {
      this.index.add(Array.from(vec));              // expects number[]
      const row = this.index.ntotal() - 1;          // ntotal() ⇒ after add
      this.rowToChunk.set(row, chunk);
    } catch (e) {
      logger.info("Error inserting chunk", inspect(chunk));
      logger.error(e);
    }
  }

  /** Add or overwrite one entity */
  async upsertEntity(ent: Entity) {
    await this.upsertChunk(entityToChunk(ent));
  }

  async upsertNarrative(turnNumber: number, narrative: string) {
    for (const chunk of narrativeToChunks(turnNumber, narrative)) {
      await this.upsertChunk(chunk);
    }
  }

  /** Remove all vectors belonging to an entity id */
  remove(chunkId: string) {
    const row = [...this.rowToChunk.entries()]
      .find(([, ch]) => ch.id === chunkId)?.[0];
    if (row === undefined) return false;

    this.index.removeIds([row]);
    this.rowToChunk.delete(row);
    return true;
  }

  /** Semantic search → entity IDs (string[]) */
  async search(query: string, k = 10) : Promise<MemoryChunk[]> {
    k = Math.min(k, this.index.ntotal() - 1);
    if (k < 1) return [];
    const qVec = await this.embed(query);
    const { labels } = this.index.search(Array.from(qVec), k);

    // labels shape = [ [row0,row1,…] ]
    return labels
      .filter((r: number) => r !== -1)
      .map((r: number) => this.rowToChunk.get(r))
      .filter((c: MemoryChunk | undefined) => c !== undefined);
  }
}
