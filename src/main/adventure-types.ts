import { MemoryGraph, MemoryGraphUpdate } from "./mcp-client";
import fs from 'fs';
import * as yaml from 'js-yaml';
import { Entity, MemoryVectorStore } from "./memory-graph";
import OpenAI from 'openai';
import { ZodType } from 'zod';

export class TurnValidationResult {
    public errors: string[] = [];

    public isFailed(): boolean {
        return this.errors.length > 0;
    }

    public isSuccess(): boolean {
        return this.errors.length === 0;
    }
}

export interface LLMCallParameters {
    llmModel: string
    maxTokens: number
    stopSequence: string | null
    jsonOutput: boolean
    schema: ZodType | null
    reasoning: OpenAI.ReasoningEffort
}

export interface AdventurePhaseConfig {
    agentName: string;
    prompts: string[];
    prefill: string;
    saveMessageToHistory: boolean;
    retryCount: number;
    llmParameters: LLMCallParameters;
}

export type ImageRole = 'player' | 'background' | 'illustration';

export interface AdventureImageUpdate {
    role: ImageRole;
    imagePrompt: string;
    negativePrompt: string;
}

export interface AdventureUserInput {
    action?: string;
    outOfCharacter?: string;
    error?: string;
    errorDetails?: string[];
}

export interface AdventureTurnFeedback {
    feedbackType: 'like' | 'dislike';
    feedbackComment: string;
}

export interface AdventureTurnInfo {
    turnNumber: number;
    fullWriterResponse: string;
    suggestedActions: string;
    userInput?: AdventureUserInput;
    illustrationType: string;
    images: AdventureImageUpdate[];
    feedback?: AdventureTurnFeedback;
    criticFeedback?: string;
}


export interface AdventureScene {
    dateTime: string;
    location: string;
    cast: string[];
    otherEntities: string[];
    description: string;
}

export interface ImagePromptParameters {
    model: string;
    characterStartPrompt: string;
    characterEndPrompt: string;
    characterNegativePrompt: string;
    itemsStartPrompt: string;
    itemsEndPrompt: string;
    itemsNegativePrompt: string;
}

export interface StoryStartingParameters {
    backstory: string;
    novelInstructions: string;
    authorStyle: string;
    firstInput: string;
    narrativeInstructions: string;
    imageInstructions: string;
    plotPlan?: string;
    entities: Entity[];
    importantEntities: string[];
    imageParameters: ImagePromptParameters;
}

export class AdventureState {
    public turns: AdventureTurnInfo[] = [];
    public lastSummarizedTurn: number = -1;
    public parameters: Record<string, string> = {};
    public importantEntities: string[] = [];
    public imagePromptParameters: ImagePromptParameters;
    public memoryGraph: MemoryGraph = new MemoryGraph();
    public fetchedEntities: Record<string, number> = {};

    private entitiesMemoryStore: MemoryVectorStore = new MemoryVectorStore();
    private narrativeMemoryStore: MemoryVectorStore = new MemoryVectorStore();

    constructor() {
        this.imagePromptParameters = {
            model: "",
            characterStartPrompt: "",
            characterEndPrompt: "",
            characterNegativePrompt: "",
            itemsStartPrompt: "",
            itemsEndPrompt: "",
            itemsNegativePrompt: "",
        };
    }

    public async initAdventure(parameters: StoryStartingParameters) {
        this.parameters["BACKSTORY"] = parameters.backstory;
        this.parameters["NOVEL_INSTRUCTIONS"] = parameters.novelInstructions;
        this.parameters["AUTHOR_STYLE"] = parameters.authorStyle;
        this.parameters["FIRST_INPUT"] = parameters.firstInput;
        this.parameters["NARRATIVE_INSTRUCTIONS"] = parameters.narrativeInstructions;
        this.parameters["IMAGE_PROMPT_INSTRUCTIONS"] = parameters.imageInstructions;
        this.parameters["PLOT_PLAN"] = parameters.plotPlan || "";
        this.importantEntities = parameters.importantEntities;
        const memoryGraphUpdate: MemoryGraphUpdate = {};
        for (const entity of parameters.entities) {
            memoryGraphUpdate[entity.id] = entity;
        }
        this.entitiesMemoryStore = new MemoryVectorStore();
        this.narrativeMemoryStore = new MemoryVectorStore();
        await this.entitiesMemoryStore.init();
        await this.narrativeMemoryStore.init();
        this.updateMemoryGraph(memoryGraphUpdate);
        this.imagePromptParameters = parameters.imageParameters;

        this.turns = [
            {
                turnNumber: 0,
                fullWriterResponse: `<response><narrative>${parameters.backstory}</narrative></response>`,
                suggestedActions: "",
                illustrationType: "",
                images: [],
            }
        ]
    }

    public getEntitiesMemoryStore(): MemoryVectorStore {
        return this.entitiesMemoryStore;
    }

    public getNarrativeMemoryStore(): MemoryVectorStore {
        return this.narrativeMemoryStore;
    }

    public serialize(): string {
        return yaml.dump({
            turns: this.turns,
            lastSummarizedTurn: this.lastSummarizedTurn,
            parameters: this.parameters,
            importantEntities: this.importantEntities,
            imagePromptParameters: this.imagePromptParameters,
            memoryGraph: this.memoryGraph,
            fetchedEntities: this.fetchedEntities,
        });
    }

    public async deserialize(yamlString: string) {
        let loadedState = yaml.load(yamlString) as any;
        this.entitiesMemoryStore = new MemoryVectorStore();
        this.narrativeMemoryStore = new MemoryVectorStore();
        await this.entitiesMemoryStore.init();
        await this.narrativeMemoryStore.init();

        if (loadedState.turns) this.turns = loadedState.turns;
        if (loadedState.lastSummarizedTurn) this.lastSummarizedTurn = loadedState.lastSummarizedTurn;
        if (loadedState.parameters) this.parameters = loadedState.parameters;
        if (loadedState.importantEntities) this.importantEntities = loadedState.importantEntities;
        if (loadedState.imagePromptParameters) this.imagePromptParameters = loadedState.imagePromptParameters;
        if (loadedState.memoryGraph) this.memoryGraph = loadedState.memoryGraph;
        if (loadedState.fetchedEntities) this.fetchedEntities = loadedState.fetchedEntities;

        for (const entity of Object.values(this.memoryGraph.entities)) {
            await this.entitiesMemoryStore.upsertEntity(entity);
        }
    }

    public updateMemoryGraph(memoryGraphUpdate: MemoryGraphUpdate): TurnValidationResult {
        let result = new TurnValidationResult();
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
            this.entitiesMemoryStore.upsertEntity(this.memoryGraph.entities[id]);
        }
        return result;
    }

    public resolvePrompt(promptPath: string) {
        let systemPrompt = fs.readFileSync(promptPath, "utf-8");

        systemPrompt = systemPrompt.replace(/{{(.*?)}}/g, (match, parameterName) => {
            if (this.parameters[parameterName]) {
                return this.parameters[parameterName];
            } else {
                return match; // Leave the placeholder as is if no replacement is found
            }
        });
        return systemPrompt;
    }

    public getLastTurn(): AdventureTurnInfo {
        return this.turns[this.turns.length - 1];
    }

    public updateImage(update: AdventureImageUpdate) {
        if (this.getLastTurn().images.some(image =>
            image.role === update.role &&
            image.imagePrompt === update.imagePrompt &&
            image.negativePrompt === update.negativePrompt)) {
            return;
        }
        this.getLastTurn().images = this.getLastTurn().images.filter(image => image.role !== update.role);
        this.getLastTurn().images.push(update);
    }

    public makeImageUpdate(role: ImageRole, prompt: string, entityType: string): AdventureImageUpdate {
        const prompts = this.imagePromptParameters;
        const startEndPrompts = (entityType === "character") ? { start: prompts.characterStartPrompt, end: prompts.characterEndPrompt } : { start: prompts.itemsStartPrompt, end: prompts.itemsEndPrompt };
        return {
            role: role,
            imagePrompt: startEndPrompts.start + prompt + startEndPrompts.end,
            negativePrompt: (entityType === "character") ? prompts.characterNegativePrompt : prompts.itemsNegativePrompt,
        }
    }

    public getParameterOrDefault(name: string, defaultValue: string) {
        return this.parameters[name] || defaultValue;
    }

    public addFetchedEntity(entityId: string) {
        if (this.memoryGraph.entities[entityId] === undefined) {
            return;
        }
        const MAX_FETCHED_ENTITIES = 20;
        const MIN_ENTITY_AGE_TO_DELETE = 2;
        
        this.fetchedEntities[entityId] = this.getLastTurn().turnNumber;
        while (this.fetchedEntities.size > MAX_FETCHED_ENTITIES) {
            const oldestEntityId = Object.keys(this.fetchedEntities).reduce((oldest, current) => {
                return current[1] < oldest[1] ? current : oldest;
            })[0];
            if (this.getLastTurn().turnNumber - this.fetchedEntities[oldestEntityId] < MIN_ENTITY_AGE_TO_DELETE) {
                break;
            }
            delete this.fetchedEntities[oldestEntityId];
        }
    }

    public getFetchedEntities(fetchAll: boolean): string {
        if (fetchAll) {
            return yaml.dump(Object.values(this.memoryGraph.entities), { lineWidth: -1 });
        }

        for (const importantEntityId of this.importantEntities) {
            if (!this.fetchedEntities[importantEntityId]) {
                this.addFetchedEntity(importantEntityId);
            }
        }
        const entities = Object.keys(this.fetchedEntities).map(entityId => this.memoryGraph.entities[entityId]).filter(entity => entity !== undefined);
        return yaml.dump(entities, { lineWidth: -1 });
    }

    public getRecentTurns(firstTurn: number, lastTurnInclusive: number): AdventureTurnInfo[] {
        let resentTurns: AdventureTurnInfo[] = [];
        for (let i = firstTurn; i <= lastTurnInclusive; i++) {
            if (i >= 0 && i < this.turns.length) {
                resentTurns.push(this.turns[i]);
            }
        }
        return resentTurns;
    }
}
