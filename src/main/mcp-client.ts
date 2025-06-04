import { OpenAI } from 'openai';
import { logger } from "./logger";
import fs from "fs";
import * as yaml from 'js-yaml';
import { Entity, MemoryVectorStore } from './memory-graph';

export type EntityMap = { [key: string]: Entity };

export class MemoryGraph {
    entities: EntityMap = {};
}

export type MemoryGraphUpdate = Record<string, Partial<Entity>>

// Openrouter API requires also "name" for some reason, idk
export interface ToolCallResult extends OpenAI.ChatCompletionToolMessageParam {
    name: string;
}

const toolsSchema: OpenAI.ChatCompletionTool[] = [
    {
        type: "function",
        function: {
            name: "search-memory",
            description: "Free‑text search across the memory. Call this whenever the narrative mentions a name or a keyword that is missing from the chat history.",
            parameters: {
                type: "object",
                properties: {
                    query: {
                        type: "string",
                        description: "String to look for"
                    }
                },
                required: ["query"]
            }
        }
    },
    {
        type: "function",
        function: {
            name: "get-related-entities",
            description: "Return all entities related to the selected entity. The entity id must match exactly. Use this function to get more detailed information about the entity.",
            parameters: {
                type: "object",
                properties: {
                    id: {
                        type: "string",
                        description: "Exact id of the entity to fetch."
                    }
                },
                required: ["id"]
            }
        }
    },
]

export class MCPClient {
    private tools: OpenAI.ChatCompletionTool[];
    private memoryGraph: MemoryGraph = new MemoryGraph();
    private memoryStore: MemoryVectorStore = new MemoryVectorStore();

    constructor() {
        this.tools = toolsSchema;
    }

    public async start() {
        await this.memoryStore.init();
        await this.loadState();
        logger.info("Started to MCP client with tools:", JSON.stringify(this.tools));
    }

    public getTools(): OpenAI.ChatCompletionTool[] {
        return this.tools;
    }

    private toToolCallResult(toolCall: OpenAI.ChatCompletionMessageToolCall, response: any): ToolCallResult {
        return {
            role: "tool",
            tool_call_id: toolCall.id,
            name: toolCall.function.name,
            content: [{ type: "text", text: JSON.stringify(response) }],
        }
    }

    public getMemoryGraph(): MemoryGraph {
        return this.memoryGraph;
    }

    public getMemoryStore(): MemoryVectorStore {
        return this.memoryStore;
    }

    async callTool(toolCall: OpenAI.ChatCompletionMessageToolCall): Promise<ToolCallResult> {
        logger.info(`Calling tool: ${JSON.stringify(toolCall)}...`)

        let toolArgs
        try {
            toolArgs = JSON.parse(toolCall.function.arguments);
        } catch (error) {
            toolArgs = {};
        }

        try {
            switch (toolCall.function.name) {
                // TODO
            }
            throw new Error(`Tool "${toolCall.function.name}" doesn't exist`);
        } catch (error) {
            console.error(`Error calling tool: ${toolCall.function.name}`, error)
            return this.toToolCallResult(toolCall, error);
        }
    }

    public updateMemoryGraph(memoryGraphUpdate: MemoryGraphUpdate): { errors: string[] } {
        let result: { errors: string[] } = { errors: [] };
        for (const id in memoryGraphUpdate) {

            if (this.memoryGraph.entities[id]) {
                this.memoryGraph.entities[id] = { ...this.memoryGraph.entities[id], ...memoryGraphUpdate[id] };
            }
            else {
                let newEntity: Entity = {
                    id: id,
                    type: "entity",
                    name: "",
                    info: "",
                }
                this.memoryGraph.entities[id] = { ...newEntity, ...memoryGraphUpdate[id] };
            }
            this.memoryStore.upsertEntity(this.memoryGraph.entities[id]);
        }
        this.saveState();
        return result;
    }

    private saveState() {
        fs.writeFileSync("saved/memory-graph.yaml", yaml.dump(this.memoryGraph));
    }

    private async loadState() {
        if (fs.existsSync("saved/memory-graph.yaml")) {
            this.memoryGraph = yaml.load(fs.readFileSync("saved/memory-graph.yaml", "utf-8")) as MemoryGraph;
            for (const entity of Object.values(this.memoryGraph.entities)) {
                await this.memoryStore.upsertEntity(entity);
            }
        }
    }
}