import { LLMClient } from "./llm-client";
import { EventEmitter } from "stream";
import * as yaml from 'js-yaml';
import { logger } from "./logger";
import { AdventureImageUpdate, AdventurePhaseConfig, AdventureScene, AdventureState, AdventureTurnInfo, AdventureUserInput, TurnValidationResult } from './adventure-types';
import { MemoryGraphUpdate } from "./mcp-client";
import { extractTurnNumber, MemoryChunk } from "./memory-graph";
import dotenv from 'dotenv';

const NEW_ADVENTURE_PROMPT_PATH = "prompts/new-adventure-prompt.txt";
const SUMMARY_PROMPT_PATH = "prompts/summarize-prompt.txt";
const HISTORY_PROMPT_PATH = "prompts/history-prompt.txt";
const MEMORY_FETCH_PROMPT_PATH = "prompts/memory-fetch-prompt.txt";
const MEMORY_FETCH_RESULT_PROMPT_PATH = "prompts/memory-fetch-result-prompt.txt";
const ANALYZE_PROMPT_PATH = "prompts/analyze-prompt.txt";
const NARRATIVE_PROMPT_PATH = "prompts/narrative-prompt.txt";
const MEMORY_UPDATE_PROMPT_PATH = "prompts/memory-update-prompt.txt";

dotenv.config();

const OPENROUTER_MODEL = process.env.OPENROUTER_MODEL;
if (!OPENROUTER_MODEL) {
    throw new Error("OPENROUTER_MODEL is not set");
}
const OPENROUTER_MODEL_MEMORY_FETCH = process.env.OPENROUTER_MODEL_MEMORY_FETCH || OPENROUTER_MODEL;
const OPENROUTER_MODEL_NARRATIVE = process.env.OPENROUTER_MODEL_NARRATIVE || OPENROUTER_MODEL;

const TURNS_TO_KEEP = 10;
const TURNS_TO_KEEP_IN_HISTORY = 4;
const TURNS_TO_SUMMARIZE = 5;

class AdventureLLMPhase {
    protected llmClient: LLMClient;
    protected config: AdventurePhaseConfig;
    protected adventureState: AdventureState;
    protected accumulatedResponse: string = "";

    constructor(llmClient: LLMClient, config: AdventurePhaseConfig, adventureState: AdventureState) {
        this.llmClient = llmClient;
        this.config = config;
        this.adventureState = adventureState;
    }

    public async startPhase(onTurnUpdated: () => void): Promise<TurnValidationResult> {
        let result = new TurnValidationResult();
        const systemPrompt = this.config.prompts.map((promptPath) => this.adventureState.resolvePrompt(promptPath)).join("");
        const previousMessages = [...this.llmClient.getMessages()];
        this.llmClient.replaceMessage(0, {
            role: "system",
            content: systemPrompt,
        });
        for (let attempt = 1; attempt <= this.config.retryCount; attempt++) {
            this.llmClient.addMessage({
                role: "assistant",
                content: this.config.prefill
            })
            this.accumulatedResponse = this.config.prefill;
            await this.llmClient.query(this.config.llmModel, this.config.maxTokens, this.config.schema, (chunk) => {
                this.addTextChunk(chunk, onTurnUpdated);
            });

            result = new TurnValidationResult();
            if (this.accumulatedResponse.length === 0) {
                result.errors.push("Response is empty");
            } else {
                try {
                    result = await this.parsePhaseResult();
                } catch (error: unknown) {
                    const errorMessage = error instanceof Error ? error.message : String(error);
                    result.errors.push("Failed to parse response: " + errorMessage);
                }
            }
            if (result.isFailed()) {
                logger.error("\n\n--- Errors found in phase result:", result.errors);
                this.llmClient.addMessage({
                    role: "user",
                    content: yaml.dump({
                        error: "Can't parse your response. The message was discarded. You have to try again. Keep attention to the required output format. Attempts left: " + (this.config.retryCount - attempt),
                        errorDetails: result.errors,
                    })
                });
                continue;
            }
            else {
                this.llmClient.replaceMessageHistory(previousMessages);
                if (this.config.saveMessageToHistory) {
                    this.llmClient.addMessage({
                        role: "assistant",
                        name: this.config.agentName,
                        content: this.accumulatedResponse,
                    });
                }
                this.finalizeHistory();
                return result;
            }
        }
        return result;
    }

    protected addTextChunk(chunk: string, onTurnUpdated: () => void) {
        this.accumulatedResponse += chunk;
    }

    protected setParameterOverride(name: string, value: string) {
        this.adventureState.parameters[name] = value;
    }

    public async parsePhaseResult(): Promise<TurnValidationResult> {
        return new TurnValidationResult();
    }

    public finalizeHistory() { }

    protected findXMLSection(response: string, sectionName: string): string | null {
        const startTag = `<${sectionName}>`;
        const endTag = `</${sectionName}>`;
        const startIndex = response.indexOf(startTag);
        const endIndex = response.indexOf(endTag, startIndex);

        if (startIndex !== -1 && endIndex !== -1) {
            return response.substring(startIndex + startTag.length, endIndex).trim();
        }
        return null;
    }
}

class AdventureLLMPhaseMemoryFetch extends AdventureLLMPhase {

    constructor(llmClient: LLMClient, adventureState: AdventureState) {
        const config: AdventurePhaseConfig = {
            llmModel: OPENROUTER_MODEL_MEMORY_FETCH,
            agentName: "Search Agent",
            prompts: [MEMORY_FETCH_PROMPT_PATH],
            maxTokens: 400,
            prefill: "<response>\n<entities>\n",
            schema: null,
            saveMessageToHistory: false,
            retryCount: 3,
        };
        super(llmClient, config, adventureState);
    }

    public async parsePhaseResult(): Promise<TurnValidationResult> {
        const result = new TurnValidationResult();
        const entitiesSection = this.findXMLSection(this.accumulatedResponse, "entities");
        const searchSection = this.findXMLSection(this.accumulatedResponse, "search");
        if (!entitiesSection) {
            result.errors.push("Can't parse <entities>. Did you forget to close the tags again?");
        }
        if (!searchSection) {
            result.errors.push("Can't parse <search>. Did you forget to close the tags again?");
        }
        if (result.isFailed()) {
            return result;
        }
        let entities: MemoryGraphUpdate = {};
        for (const entityId of entitiesSection!.split("\n")) {
            const entity = this.adventureState.memoryGraph.entities[entityId.trim()]
            if (entity && !this.adventureState.fetchedEntities.has(entityId)) {
                entities[entity.id] = entity;
                this.adventureState.fetchedEntities.add(entityId);
            }
        }
        this.setParameterOverride("SEARCHED_ENTITIES", yaml.dump(entities, { lineWidth: -1 }));

        let existingChunks = new Set<string>();
        let searchResults: { query: string, results: any[] }[] = [];
        for (const query of searchSection!.split("\n")) {
            searchResults.push({ query: query.trim(), results: await this.search(query.trim(), existingChunks) });
        }
        this.setParameterOverride("SEARCHED_RESULTS", yaml.dump(searchResults, { lineWidth: -1 }));
        return result;
    }

    private async search(query: string, existingChunks: Set<string>): Promise<any[]> {
        const DESIRED_RESULT_COUNT = 10;

        let foundChunks = await this.adventureState.getMemoryStore().search(query, DESIRED_RESULT_COUNT * 2);
        foundChunks = foundChunks.filter((chunk) => {
            if (existingChunks.has(chunk.id)) {
                return false;
            }
            if (chunk.meta?.type === "entity" && this.adventureState.fetchedEntities.has(chunk.id)) {
                return false;
            }
            if (chunk.meta?.type === "narrative") {
                const turnNumber = extractTurnNumber(chunk.id);
                if (turnNumber && this.adventureState.visibleTurnNumbers.has(turnNumber)) {
                    return false;
                }
            }
            return true;
        }).slice(0, DESIRED_RESULT_COUNT);
        foundChunks.forEach((chunk) => existingChunks.add(chunk.id));
        return foundChunks.map((chunk) => chunk.meta?.type === "entity" ? yaml.load(chunk.text) : chunk.text);
    }

    public finalizeHistory() {
        // Trick here - we don't save the memory fetch message, but put the result in reserved place
        for (const message of this.llmClient.getMessages()) {
            if (message.role === "user" && message.name === "Memory Provider") {
                message.content = this.adventureState.resolvePrompt(MEMORY_FETCH_RESULT_PROMPT_PATH);
            }
        }
    }
}

class AdventureLLMPhaseAnalyze extends AdventureLLMPhase {

    constructor(llmClient: LLMClient, adventureState: AdventureState) {
        const config: AdventurePhaseConfig = {
            llmModel: OPENROUTER_MODEL!,
            agentName: "Analyzer Agent",
            prompts: [ANALYZE_PROMPT_PATH],
            maxTokens: 1000,
            prefill: "<response>\n<analysis>\n",
            schema: null,
            saveMessageToHistory: true,
            retryCount: 3,
        };
        super(llmClient, config, adventureState);
    }

    public async parsePhaseResult(): Promise<TurnValidationResult> {
        const result = new TurnValidationResult();

        const analysisSection = this.findXMLSection(this.accumulatedResponse, "analysis");
        if (!analysisSection) {
            result.errors.push("Can't parse <analysis>. Likely due to malformed XML.");
        }
        const sceneSection = this.findXMLSection(this.accumulatedResponse, "scene");
        if (!sceneSection) {
            result.errors.push("Can't parse <scene>. Likely due to malformed XML.");
        }
        const narrativePlanSection = this.findXMLSection(this.accumulatedResponse, "narrativePlan");
        if (!narrativePlanSection || narrativePlanSection.trim().length === 0) {
            result.errors.push("Can't parse <narrativePlan>. Likely due to malformed XML.");
        }
        if (result.errors.length > 0) {
            return result;
        }
        this.adventureState.getLastTurn().analysis = analysisSection!;
        this.adventureState.getLastTurn().scene = sceneSection!;
        this.adventureState.getLastTurn().narrativePlan = narrativePlanSection!;
        return result;
    }
}

class AdventureLLMPhaseNarrative extends AdventureLLMPhase {

    constructor(llmClient: LLMClient, adventureState: AdventureState) {
        const config: AdventurePhaseConfig = {
            llmModel: OPENROUTER_MODEL_NARRATIVE,
            agentName: "Writer Agent",
            prompts: [NARRATIVE_PROMPT_PATH],
            maxTokens: 3000,
            prefill: "<response>\n",
            schema: null,
            saveMessageToHistory: true,
            retryCount: 2,
        };
        super(llmClient, config, adventureState);
    }

    public async parsePhaseResult(): Promise<TurnValidationResult> {
        const result = new TurnValidationResult();
        const narrativeSection = this.findXMLSection(this.accumulatedResponse, "narrative");
        if (!narrativeSection) {
            result.errors.push("Can't parse <narrative>. Likely due to malformed XML.");
        }
        if (narrativeSection?.trim().length === 0) {
            result.errors.push("Can't parse <narrative>. Empty narrative.");
        }
        const suggestedActionsSection = this.findXMLSection(this.accumulatedResponse, "suggestedActions");
        if (!suggestedActionsSection) {
            result.errors.push("Can't parse <suggestedActions>. Likely due to malformed XML.");
        }
        if (result.isFailed()) {
            return result;
        }
        this.adventureState.getLastTurn().narrative = narrativeSection!.trim();
        this.adventureState.getLastTurn().suggestedActions = suggestedActionsSection!.trim();
        return result;
    }

    protected addTextChunk(chunk: string, onTurnUpdated: () => void) {
        this.accumulatedResponse += chunk;
        const narrativeSection = this.findResponsePartialSection(this.accumulatedResponse, "narrative");
        if (narrativeSection) {
            this.adventureState.getLastTurn().narrative = narrativeSection;
            onTurnUpdated();
        }
    }

    protected findResponsePartialSection(response: string, sectionName: string): string | null {
        const fullSection = this.findXMLSection(response, sectionName);
        if (fullSection) {
            return fullSection;
        }
        const startTag = `<${sectionName}>`;
        const startIndex = response.indexOf(startTag);
        if (startIndex !== -1) {
            const endSectionIndex = response.indexOf("<", startIndex + startTag.length);
            return response.substring(startIndex + startTag.length, endSectionIndex !== -1 ? endSectionIndex : undefined);
        }
        return null;
    }
}

interface LLMResponseMemoryUpdate {
    turnSelfAssesment: any;
    newEntities: MemoryGraphUpdate;
    entityUpdates: MemoryGraphUpdate;
    backgroundPrompt: string;
    illustrationId: string;
    illustrationPrompt: string;
    playerPortraitPrompt: string;
}


class AdventureLLMPhaseMemoryUpdate extends AdventureLLMPhase {

    constructor(llmClient: LLMClient, adventureState: AdventureState) {
        const config: AdventurePhaseConfig = {
            llmModel: OPENROUTER_MODEL!,
            agentName: "Memory Agent",
            prompts: [MEMORY_UPDATE_PROMPT_PATH],
            maxTokens: 2000,
            prefill: "# memory update start\n",
            schema: null,
            saveMessageToHistory: false,
            retryCount: 5,
        };
        super(llmClient, config, adventureState);
    }

    public async parsePhaseResult(): Promise<TurnValidationResult> {
        const result = new TurnValidationResult();
        const response = yaml.load(this.accumulatedResponse) as LLMResponseMemoryUpdate;
        if (response.newEntities) {
            const existingNewEntities = Object.keys(response.newEntities).filter((key) => this.adventureState.memoryGraph.entities[key]);
            if (existingNewEntities.length > 0) {
                result.errors.push(`'newEntities' section contains entities already present in memory: ${existingNewEntities.join(", ")}`);
            }
        }
        if (response.entityUpdates) {
            for (const entityId of Object.keys(response.entityUpdates)) {
                const entity = this.adventureState.memoryGraph.entities[entityId];
                if (!entity) {
                    result.errors.push(`Entity ${entityId} must be added to memory before updating.`);
                }
                else if (response.entityUpdates[entityId].info) {
                    result.errors.push(`Info of ${entityId} cannot be changed. Use 'state' instead`);
                } else if (response.entityUpdates[entityId].secret) {
                    result.errors.push(`Secret of ${entityId} cannot be changed. Use 'state' instead`);
                }
            }
        }
        if (!response.illustrationId || (this.adventureState.memoryGraph.entities[response.illustrationId] == null && response.newEntities[response.illustrationId] == null)) {
            result.errors.push(`illustration ${response.illustrationId} is not a valid entity.`);
        }
        if (!response.backgroundPrompt) {
            result.errors.push("backgroundPrompt is missing in the response.");
        }
        if (!response.illustrationPrompt) {
            result.errors.push("illustrationPrompt is missing in the response.");
        }
        if (!response.playerPortraitPrompt) {
            result.errors.push("playerPortraitPrompt is missing in the response.");
        }
        if (result.isFailed()) {
            return result;
        }
        if (response.newEntities) {
            result.errors.push(...this.updateMemory(response.newEntities).errors);
        }
        if (response.entityUpdates) {
            result.errors.push(...this.updateMemory(response.entityUpdates).errors);
        }
        if (result.isFailed()) {
            return result;
        }
        this.adventureState.getLastTurn().illustrationId = response.illustrationId;
        this.setParameterOverride("PREVIOUS_BACKGROUND_PROMPT", response.backgroundPrompt.trim());
        this.setParameterOverride("PREVIOUS_PLAYER_PROMPT", response.playerPortraitPrompt.trim());
        result.errors.push(...this.updateTurnImages(response).errors);
        return result;
    }

    private updateMemory(memoryUpdates: MemoryGraphUpdate): TurnValidationResult {
        const result = new TurnValidationResult()
        try {
            result.errors.push(...this.adventureState.updateMemoryGraph(memoryUpdates).errors);
        } catch (error) {
            result.errors.push(`invalid memory update request: ${error}`);
        }
        return result;
    }

    private updateTurnImages(response: LLMResponseMemoryUpdate) {
        const result = new TurnValidationResult();
        const memoryGraph = this.adventureState.memoryGraph;
        const settingPrompt = response.backgroundPrompt.trim();
        try {
            this.adventureState.updateImage(this.adventureState.makeImageUpdate("background", settingPrompt, "location"));
        } catch (error) {
            result.errors.push("Error generating background image: " + error);
        }
        try {
            const playerPrompt = this.adventureState.makeImageUpdate("player", response.playerPortraitPrompt + ", located in " + settingPrompt, "character");
            this.adventureState.updateImage(playerPrompt);
        } catch (error) {
            result.errors.push("Error generating player image: " + error);
        }
        try {
            const illustrationEntity = memoryGraph.entities[response.illustrationId];
            if (!illustrationEntity) {
                throw new Error(`Illustration entity "${response.illustrationId}" not found in memory graph.`);
            }
            const illustrationPrompt = this.adventureState.makeImageUpdate("illustration", response.illustrationPrompt + ", located in " + settingPrompt, illustrationEntity.type);
            this.adventureState.updateImage(illustrationPrompt);
        } catch (error) {
            result.errors.push("Error generating illustration image: " + error);
        }
        return result;
    }
}

interface LLMResponseSummary {
    summary: string;
    analysis: string;
    plotPlan: string;
    userProfile: string;
}

class AdventureLLMPhaseSummary extends AdventureLLMPhase {

    constructor(llmClient: LLMClient, adventureState: AdventureState) {
        const config: AdventurePhaseConfig = {
            llmModel: OPENROUTER_MODEL!,
            agentName: "Summary Agent",
            prompts: [SUMMARY_PROMPT_PATH],
            maxTokens: 2000,
            prefill: "# summary start\n",
            schema: null,
            saveMessageToHistory: false,
            retryCount: 5,
        };
        super(llmClient, config, adventureState);
    }

    public async parsePhaseResult(): Promise<TurnValidationResult> {
        const result = new TurnValidationResult();
        const response = yaml.load(this.accumulatedResponse) as LLMResponseSummary;
        this.adventureState.parameters["PLOT_PLAN"] = response.plotPlan;
        this.adventureState.parameters["USER_PROFILE"] = response.userProfile;
        this.adventureState.parameters["SUMMARY_ANALYSIS"] = response.analysis;
        logger.info("Summary:", response.summary);
        response.summary = this.adventureState.getParameterOrDefault("STORY_ARCHIVE", "") + response.summary;
        this.adventureState.parameters["STORY_ARCHIVE"] = response.summary;
        this.adventureState.lastSummarizedTurn = this.adventureState.turns.length - 1;
        return result;
    }
}

export class AdventureLLMRequest {
    private llmClient: LLMClient;
    private adventureState: AdventureState;
    private runningPhase: AdventureLLMPhase | null = null;

    private eventEmitter: EventEmitter = new EventEmitter();

    constructor(llmClient: LLMClient, adventureState: AdventureState) {
        this.llmClient = llmClient;
        this.adventureState = adventureState;
    }

    public onTurnUpdated(callback: (update: AdventureTurnInfo) => void) {
        this.eventEmitter.on("turn-updated", callback);
    }

    private getRecentTurns(firstTurn: number, lastTurnInclusive: number): AdventureTurnInfo[] {
        let resentTurns: AdventureTurnInfo[] = [];
        for (let i = firstTurn; i <= lastTurnInclusive; i++) {
            if (i >= 0 && i < this.adventureState.turns.length) {
                resentTurns.push(this.adventureState.turns[i]);
            }
        }
        return resentTurns;
    }

    private fetchRecentTurnNarratives(firstTurn: number, lastTurnInclusive: number): string {
        let recentTurns = this.getRecentTurns(firstTurn, lastTurnInclusive);
        let recentTurnsText = "";
        for (const turn of recentTurns) {
            this.adventureState.visibleTurnNumbers.add(turn.turnNumber);
            recentTurnsText += `--Turn ${turn.turnNumber}:\n`;
            if (turn.userInput) {
                recentTurnsText += `>${yaml.dump(turn.userInput, { lineWidth: -1 })}\n`;
            }
            recentTurnsText += turn.narrative;
            recentTurnsText += "\n";
            if (turn.feedback) {
                recentTurnsText += `Feedback: ${yaml.dump(turn.feedback, { lineWidth: -1 })}\n`;
            }
        }
        return recentTurnsText;
    }

    public async performTurn(userInput: AdventureUserInput): Promise<TurnValidationResult> {
        logger.info("\n\n--- Starting new turn with message:\n", yaml.dump(userInput));
        const adventureStateBackup = this.adventureState.serialize();
        const firstUnarchivedTurn = this.adventureState.turns.length - TURNS_TO_KEEP;
        const firstHistoryTurn = this.adventureState.turns.length - TURNS_TO_KEEP_IN_HISTORY;
        const lastHistoryTurn = this.adventureState.turns.length - 1;
        logger.info(`Turns in first prompt: ${firstUnarchivedTurn}-${firstHistoryTurn - 1}`);
        logger.info(`Turns in history: ${firstHistoryTurn}-${lastHistoryTurn}`);

        const turnNumber = this.adventureState.turns.length;
        this.adventureState.turns.push({
            turnNumber: turnNumber,
            analysis: "",
            scene: "",
            narrativePlan: "",
            narrative: "LLM is cooking",
            suggestedActions: "",
            images: [],
            userInput: userInput,
            illustrationId: "",
        });
        this.eventEmitter.emit("turn-updated", this.adventureState.getLastTurn());
        this.adventureState.visibleTurnNumbers = new Set<number>();
        this.adventureState.parameters["RECENT_TURNS"] = this.fetchRecentTurnNarratives(firstUnarchivedTurn, firstHistoryTurn - 1);
        this.adventureState.parameters["TURN_NUMBER"] = turnNumber.toString();
        this.adventureState.parameters["REFMAP"] = Object.values(this.adventureState.memoryGraph.entities).map((e) => `${e.id}: ${e.name}`).join("\n");
        this.adventureState.parameters["EXISTING_ENTITY_IDS"] = Object.keys(this.adventureState.memoryGraph.entities).join(",");
        this.adventureState.parameters["SEARCHED_ENTITIES"] = "";
        this.adventureState.parameters["SEARCHED_RESULTS"] = "";
        this.preFetchMemory(false); // TODO: fetch all entities if the turn is a new adventure
        // if (!await this.runPhase(AdventurePhase.Init, {})) {
        //     return;
        // }
        // this.llmClient.clearMessageHistory();

        this.llmClient.clearMessageHistory();
        this.llmClient.addMessage({
            role: "user",
            name: "History Provider",
            content: this.adventureState.resolvePrompt(HISTORY_PROMPT_PATH),
        });
        this.llmClient.addMessage({
            role: "user",
            name: "Memory Provider",
            content: this.adventureState.resolvePrompt(MEMORY_FETCH_RESULT_PROMPT_PATH),
        });
        for (const turn of this.getRecentTurns(firstHistoryTurn, lastHistoryTurn)) {
            this.adventureState.visibleTurnNumbers.add(turn.turnNumber);
            if (turn.userInput) {
                this.llmClient.addMessage({
                    role: "user",
                    name: "Player",
                    content: yaml.dump(turn.userInput, { lineWidth: -1 }),
                });
            }
            this.llmClient.addMessage({
                role: "assistant",
                name: "Analyzer Agent",
                content: `<response>\n<analysis>${turn.analysis}</analysis>\n<scene>${turn.scene}</scene>\n<narrativePlan>${turn.narrativePlan || ""}</narrativePlan></response>`,
            });
            this.llmClient.addMessage({
                role: turn.turnNumber > 0 ? "assistant" : "user",
                name: turn.turnNumber > 0 ? "Writer Agent" : "History Provider",
                content: `<response>\n<narrative>${turn.narrative}</narrative>\n<suggestedActions>${turn.suggestedActions}</suggestedActions>\n</response>`,
            });
            if (turn.feedback) {
                this.llmClient.addMessage({
                    role: "user",
                    name: "Player",
                    content: yaml.dump(turn.feedback, { lineWidth: -1 }),
                });
            }
        }
        this.llmClient.addMessage({
            role: "user",
            name: "Player",
            content: yaml.dump(userInput, { lineWidth: -1 }),
        });

        let phases: AdventureLLMPhase[] = [];
        if (this.adventureState.turns.length > 2) {
            phases.push(new AdventureLLMPhaseMemoryFetch(this.llmClient, this.adventureState))
        }

        phases.push(...[
            new AdventureLLMPhaseAnalyze(this.llmClient, this.adventureState),
            new AdventureLLMPhaseNarrative(this.llmClient, this.adventureState),
            new AdventureLLMPhaseMemoryUpdate(this.llmClient, this.adventureState),
        ])

        let result = new TurnValidationResult();
        for (const phase of phases) {
            result = await this.runPhase(phase);
            if (result.isFailed()) {
                break;
            }
        }

        this.llmClient.clearMessageHistory();
        // Run summary every TURNS_TO_SUMMARIZE turns
        if (result.isSuccess() && this.adventureState.turns.length - this.adventureState.lastSummarizedTurn > TURNS_TO_SUMMARIZE) {
            this.llmClient.addMessage({
                role: "user",
                name: "History Provider",
                content: this.adventureState.resolvePrompt(HISTORY_PROMPT_PATH),
            });
            this.adventureState.parameters["TURNS_TO_SUMMARIZE"] = this.fetchRecentTurnNarratives(this.adventureState.lastSummarizedTurn + 1, this.adventureState.turns.length - 1);
            result = await this.runPhase(new AdventureLLMPhaseSummary(this.llmClient, this.adventureState));
            delete this.adventureState.parameters["TURNS_TO_SUMMARIZE"];
            this.llmClient.clearMessageHistory();
        }

        if (result.isFailed()) {
            logger.error("--- Turn failed with errors:\n", result.errors, "\n\n");

            await this.adventureState.deserialize(adventureStateBackup);
            this.eventEmitter.emit("turn-updated", this.adventureState.getLastTurn());
        }
        delete this.adventureState.parameters["RECENT_TURNS"];
        delete this.adventureState.parameters["FETCHED_ENTITIES"];
        delete this.adventureState.parameters["REFMAP"];
        delete this.adventureState.parameters["SEARCHED_ENTITIES"];
        delete this.adventureState.parameters["SEARCHED_RESULTS"];
        return result;
    }

    private async runPhase(phase: AdventureLLMPhase): Promise<TurnValidationResult> {
        let result = new TurnValidationResult();
        this.runningPhase = phase;
        try {
            result = await phase.startPhase(() => {
                this.eventEmitter.emit("turn-updated", this.adventureState.getLastTurn());
            });
            if (result.isFailed()) {
                return result;
            }
        } catch (error: unknown) {
            const errorMessage = error instanceof Error ? error.message : String(error);
            result.errors.push(errorMessage);
            return result;
        }
        if (result.isSuccess()) {
            this.eventEmitter.emit("turn-updated", this.adventureState.getLastTurn());
        }
        return result;
    }

    private preFetchMemory(fetchAll: boolean) {
        if (fetchAll) {
            let allEntities: MemoryGraphUpdate = {};
            for (const entity of Object.values(this.adventureState.memoryGraph.entities)) {
                allEntities[entity.id] = entity;
            }
            this.adventureState.parameters["FETCHED_ENTITIES"] = yaml.dump(allEntities, { lineWidth: -1 });
            return;
        }

        let entityIds: string[] = [...this.adventureState.importantEntities];
        this.adventureState.fetchedEntities = new Set(entityIds);
        entityIds = Array.from(this.adventureState.fetchedEntities); // Remove duplicates

        let entities: MemoryGraphUpdate = {};
        for (const entityId of entityIds) {
            if (this.adventureState.memoryGraph.entities[entityId]) {
                entities[entityId] = this.adventureState.memoryGraph.entities[entityId];
            }
        }
        this.adventureState.parameters["FETCHED_ENTITIES"] = yaml.dump(entities, { lineWidth: -1 });
    }
}
