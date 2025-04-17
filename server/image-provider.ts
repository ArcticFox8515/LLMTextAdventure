/**
 * Raw JSON workflow produced by ComfyUI’s “Save (API Format)” button.
 * The only runtime condition we check is that it looks like JSON.
 */
export type WorkflowJSON = string;

/** Parameters accepted by `generatePortrait`. */
export interface GeneratorOptions {
  model: string;
  prompt: string;
  negativePrompt: string;
}

/** Minimal interface consumed by the adventure engine. */
export interface ImageProvider {
  generateImage(opts: GeneratorOptions): Promise<Buffer[]>;
}

/* ------------------------------------------------------------------ */
/*  Implementation                                                    */
/* ------------------------------------------------------------------ */

import fetch from "node-fetch";
import WebSocket from "ws";
import fs from 'fs';
import { v4 as uuid } from "uuid";
import path from 'path';
import crypto from 'crypto';
import { logger } from "./logger";
import dotenv from 'dotenv';


dotenv.config();

const WORKFLOW_PATH = "prompts/image-generator-prompt.json";
const SERVER_URL = process.env.COMFYUI_API_URL;
const IMAGE_CACHE_DIR = "saved/images";
const IMAGE_CACHE_SIZE = 100;

if (!fs.existsSync(IMAGE_CACHE_DIR)) {
  fs.mkdirSync(IMAGE_CACHE_DIR, { recursive: true });
}

export class ComfyProvider implements ImageProvider {
  private readonly server: URL;
  private readonly workflow: string;

  constructor() {
    this.workflow = fs.readFileSync(WORKFLOW_PATH, "utf-8");
    this.server = new URL(SERVER_URL || "http://localhost:8188");
  }

  async generateImage({ model, prompt, negativePrompt }: GeneratorOptions): Promise<Buffer[]> {
    const seed = Math.floor(Math.random() * 2 ** 32);
    const clientId = uuid();
    // Use JSON.stringify to properly escape strings for JSON
    const escapedPrompt = JSON.stringify(prompt).slice(1, -1); // Remove the surrounding quotes
    const escapedNegativePrompt = JSON.stringify(negativePrompt).slice(1, -1); // Remove the surrounding quotes

    const wf = this.workflow
      .replaceAll("${MODEL}", model)
      .replaceAll("${PROMPT}", escapedPrompt)
      .replaceAll("${NEGATIVE_PROMPT}", escapedNegativePrompt)
      .replaceAll("${SEED}", String(seed));

    try {
      const promptRes = await fetch(new URL("/prompt", this.server).toString(), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: JSON.parse(wf), client_id: clientId })
      });
      const { prompt_id } = (await promptRes.json()) as { prompt_id: string };
      return [await this.waitForImage(clientId, prompt_id)];
    } catch (error) {
      console.error("Error generating image:", error);
      return [Buffer.alloc(0)]; // Return an empty buffer on error
    }
  }

  /* ---------------- private helpers -------------------------------- */

  private waitForImage(clientId: string, promptId: string): Promise<Buffer> {
    const wsUrl = `${this.server.protocol === "https:" ? "wss" : "ws"}://${this.server.host}/ws?clientId=${clientId}`;
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(wsUrl);
      ws.once("error", reject);
      ws.on("message", async data => {
        const evt = JSON.parse(data.toString());
        if (evt.type === "executing" && evt.data.node === null) {
          ws.close();
          try {
            resolve(await this.fetchFirstImage(promptId));
          } catch (e) {
            reject(e);
          }
        }
      });
    });
  }

  private async fetchFirstImage(promptId: string): Promise<Buffer> {
    const histUrl = new URL(`/history/${promptId}`, this.server).toString();
    const hist = await fetch(histUrl).then(r => r.json());
    const firstOut = hist[promptId].outputs;
    const firstNode = firstOut[Object.keys(firstOut)[0]];
    const meta = firstNode.images[0];
    const imgUrl = new URL("/view", this.server);
    imgUrl.searchParams.set("filename", meta.filename);
    imgUrl.searchParams.set("subfolder", meta.subfolder);
    imgUrl.searchParams.set("type", meta.type);
    return fetch(imgUrl.toString()).then(r => r.buffer());
  }
}

export class ImageGenerator {
  private provider: ImageProvider;

  constructor(provider = new ComfyProvider()) {
    this.provider = provider;
  }

  public async generateImage(prompt: GeneratorOptions): Promise<Buffer[]> {
    const hash = crypto.createHash('sha256').update(JSON.stringify(prompt)).digest('hex');
    const files = fs.readdirSync(IMAGE_CACHE_DIR).filter(file => file.startsWith(hash));

    if (files.length > 0) {
      return files.map(file => fs.readFileSync(path.join(IMAGE_CACHE_DIR, file)));
    }

    logger.info(`New image prompt: '${prompt.prompt}'`);
    const images = await this.provider.generateImage(prompt);

    images.forEach((image, index) => {
      const fileName = `${hash}-${index}.png`;
      fs.writeFileSync(path.join(IMAGE_CACHE_DIR, fileName), image);
    });

    // Clean up older files if cache exceeds IMAGE_CACHE_SIZE
    const allFiles = fs.readdirSync(IMAGE_CACHE_DIR)
      .map(file => ({
        name: file,
        time: fs.statSync(path.join(IMAGE_CACHE_DIR, file)).mtime.getTime()
      }))
      .sort((a, b) => a.time - b.time);

    while (allFiles.length > IMAGE_CACHE_SIZE) {
      const oldestFile = allFiles.shift();
      if (oldestFile) {
        fs.unlinkSync(path.join(IMAGE_CACHE_DIR, oldestFile.name));
      }
    }

    return images;
  }

  public clearImage(prompt: GeneratorOptions): void {
    const hash = crypto.createHash('sha256').update(JSON.stringify(prompt)).digest('hex');
    const files = fs.readdirSync(IMAGE_CACHE_DIR).filter(file => file.startsWith(hash));
    files.forEach(file => fs.unlinkSync(path.join(IMAGE_CACHE_DIR, file)));
  }
}