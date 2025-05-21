import * as faiss from "faiss-node";
import { logger } from "./logger";
import { inspect } from "util";
import * as yaml from 'js-yaml';
import { v4 as uuid } from "uuid";
import { info } from "console";

/* ───────── Your existing domain types ───────── */
export type EntityRelation = [string, string];

export interface Entity {
  id: string;
  type: string;
  name: string;
  brief?: string;
  appearance?: string;
  clothes?: string;
  info?: string;
  secret?: string;
  state?: string;
}

export interface MemoryChunk {
  chunkId: string;
  text: string;
  meta: { 
    type: string;    
    paragraphId?: [number, number];
  };
}

export interface MemorySearchResult extends MemoryChunk {
  distance: number;
}

const entityToChunk = (e: Entity): MemoryChunk => ({
  chunkId: e.id,
  text: yaml.dump({
    info: e.info,
    secret: e.secret,
  }),
  meta: { type: "entity" }
});

function narrativeToChunks(turnNumber: number, narrative: string): MemoryChunk[] {
  const recommendedLength = 1200;
  const minLength = 400;
  const maxLength = 2000;

  const paragraphs = narrative.split(/\n+/).map(chunk => chunk.trim());
  const chunks: string[] = [];

  let currentChunk: string[] = [];
  let currentLength = 0;
  for (const paragraph of paragraphs) {
    if (currentLength + paragraph.length > maxLength && currentChunk.length > 0) {
      chunks.push(currentChunk.join("\n"));
      currentChunk = [paragraph];
      currentLength = paragraph.length;
    }
    else {
      currentChunk.push(paragraph);
      currentLength += paragraph.length;
      if (currentLength > recommendedLength) {
        chunks.push(currentChunk.join("\n"));
        currentChunk = [];
        currentLength = 0;
      }
    }
  }

  if (currentChunk.length > 0) {
    if (chunks.length == 0 || currentLength >= minLength) {
      chunks.push(currentChunk.join("\n"));
    } else {
      chunks[chunks.length - 1] += currentChunk.join("\n");
    }
  }

  return chunks.map((chunk, index) => ({
    chunkId: `narrative-${turnNumber}-${index}`,
    text: chunk,
    meta: { 
      type: "narrative",
      paragraphId: [turnNumber, index],
    }
  }));
}


/* ───────── Vector store ───────── */
export class MemoryVectorStore {
  private dim = 384;
  private index = new faiss.IndexFlatL2(this.dim);

  /** Mean‑pooled + L2‑norm embedder will be set in `init()` */
  private embed!: (txt: string) => Promise<Float32Array>;

  private rowToChunk = new Map<number, MemoryChunk>(); // FAISS row → chunk
  private knownTurns = new Set<number>();

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
    this.knownTurns.add(turnNumber);
  }

  isTurnKnown(turnNumber: number) {
    return this.knownTurns.has(turnNumber);
  }

  /** Remove all vectors belonging to an entity id */
  remove(chunkId: string) {
    const row = [...this.rowToChunk.entries()]
      .find(([, ch]) => ch.chunkId === chunkId)?.[0];
    if (row === undefined) return false;

    this.index.removeIds([row]);
    this.rowToChunk.delete(row);
    return true;
  }

  async search(query: string, topK: number): Promise<MemorySearchResult[]> {
    topK = Math.min(topK, this.index.ntotal() - 1);
    if (topK < 1) return [];
    const qVec = await this.embed(query);
    const searchResult = this.index.search(Array.from(qVec), topK);

    const result: MemorySearchResult[] = [];
    for (let i = 0; i < searchResult.labels.length; i++) {
      if (searchResult.labels[i] === -1) continue;
      const chunk = this.rowToChunk.get(searchResult.labels[i]);
      if (!chunk) {
        continue;
      }
      result.push({
        ...chunk,
        distance: searchResult.distances[i],
      });
    }
    return result;
  }

  async searchMultiple(queries: string[], topK: number, chunkIdsToExclude: string[] = []): Promise<MemorySearchResult[]> {
    const allResults: MemorySearchResult[] = [];
    for (const query of queries) {
      const searchResults = await this.search(query, topK + chunkIdsToExclude.length);
      allResults.push(...searchResults.filter(result => !chunkIdsToExclude.includes(result.chunkId)));
    }
    allResults.sort((a, b) => a.distance - b.distance); // need sorting before deduplication to keep the closest results
    const uniqueResults = new Map<string, MemorySearchResult>();
    for (const result of allResults) {
      if (!uniqueResults.has(result.chunkId)) {
        uniqueResults.set(result.chunkId, result);
      }
    }
    return [...uniqueResults.values()].sort((a, b) => a.distance - b.distance).slice(0, topK);
  }
}
