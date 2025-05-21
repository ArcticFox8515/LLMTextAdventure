import { LLMClient } from "./llm-client";
import { EventEmitter } from "stream";
import * as yaml from 'js-yaml';
import { logger } from "./logger";
import { AdventureImageUpdate, AdventurePhaseConfig, AdventureScene, AdventureState, AdventureTurnInfo, AdventureUserInput, TurnValidationResult } from './adventure-types';
import { MemoryGraphUpdate } from "./mcp-client";
import { Entity, MemoryChunk } from "./memory-graph";
import dotenv from 'dotenv';
import { z } from "zod";
import { on } from "events";

const NEW_ADVENTURE_PROMPT_PATH = "prompts/new-adventure-prompt.txt";
const SUMMARY_PROMPT_PATH = "prompts/summarize-prompt.txt";
const HISTORY_PROMPT_PATH = "prompts/history-prompt.txt";
const MEMORY_FETCH_PROMPT_PATH = "prompts/memory-fetch-prompt.txt";
const MEMORY_FETCH_RESULT_PROMPT_PATH = "prompts/memory-fetch-result-prompt.txt";
const ANALYZE_PROMPT_PATH = "prompts/analyze-prompt.txt";
const NARRATIVE_PROMPT_PATH = "prompts/narrative-prompt.txt";
const ASSISTANT_PROMPT_PATH = "prompts/assistant-prompt.txt";
const CRITIC_PROMPT_PATH = "prompts/critic-prompt.txt";

dotenv.config();

const OPENROUTER_MODEL = process.env.OPENROUTER_MODEL;
if (!OPENROUTER_MODEL) {
    throw new Error("OPENROUTER_MODEL is not set");
}
const OPENROUTER_MODEL_MEMORY_FETCH = process.env.OPENROUTER_MODEL_MEMORY_FETCH || OPENROUTER_MODEL;
const OPENROUTER_MODEL_NARRATIVE = process.env.OPENROUTER_MODEL_NARRATIVE || OPENROUTER_MODEL;
const OPENROUTER_MODEL_ASSISTANT = process.env.OPENROUTER_MODEL_ASSISTANT || OPENROUTER_MODEL;

const TURNS_TO_KEEP = 8;
const TURNS_TO_KEEP_IN_HISTORY = 4;
const TURNS_TO_SUMMARIZE = 5;


export function findXMLSection(response: string, sectionName: string, turnResult: TurnValidationResult): string | null {
    const startTag = `<${sectionName}>`;
    const endTag = `</${sectionName}>`;
    const startIndex = response.indexOf(startTag);
    const endIndex = response.indexOf(endTag, startIndex);

    if (startIndex !== -1 && response.indexOf(startTag, startIndex + startTag.length) !== -1) {
        turnResult.errors.push(`The tag "${sectionName}" appears multiple times in the answer`);
    }

    if (startIndex !== -1 && endIndex !== -1) {
        return response.substring(startIndex + startTag.length, endIndex).trim();
    }
    turnResult.errors.push(`Failed to find "${sectionName}" section in the answer`);
    return null;
}

export function findResponsePartialSection(response: string, sectionName: string): string | null {
    const fullSection = findXMLSection(response, sectionName, new TurnValidationResult());
    if (fullSection) {
        return fullSection;
    }
    const startTag = `<${sectionName}>`;
    const startIndex = response.indexOf(startTag);
    if (startIndex !== -1) {
        const endSectionIndex = response.indexOf(`</${sectionName}>`, startIndex + startTag.length);
        return response.substring(startIndex + startTag.length, endSectionIndex !== -1 ? endSectionIndex : undefined);
    }
    return null;
}

export function getTurnNarrative(turn: AdventureTurnInfo, isPartial: boolean): string {
    if (isPartial) {
        return findResponsePartialSection(turn.fullWriterResponse || "", "narrative") || "";
    }
    return findXMLSection(turn.fullWriterResponse || "", "narrative", new TurnValidationResult()) || "";
}

function cleanJSONResponse(response: string): string {
    return response
        .replace(/^```json\n?/, '')
        .replace(/\n?```$/, '')
        .trim();
}

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
        this.praparePhase();
        this.llmClient.replaceMessage(0, {
            role: "system",
            content: systemPrompt,
        });
        for (let attempt = 1; attempt <= this.config.retryCount; attempt++) {
            if (this.config.prefill.length > 0) {
                this.llmClient.addMessage({
                    role: "assistant",
                    content: this.config.prefill
                })
            }
            this.accumulatedResponse = this.config.prefill;
            await this.llmClient.query(this.config.llmParameters, (chunk) => {
                this.addTextChunk(chunk, onTurnUpdated);
            });
            if (this.config.llmParameters.stopSequence && !this.accumulatedResponse.includes(this.config.llmParameters.stopSequence)) {
                this.accumulatedResponse += this.config.llmParameters.stopSequence;
            }

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
                        error: "Wrong output format. The message was discarded. Re-generate the response. Don't perform reasoning or write free-form text, output only the response.",
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

    protected praparePhase() { }

    public finalizeHistory() { }
}

interface MemoryFetchResult {
    entities: string[];
    search: string[];
}

const memoryFetchResultSchema = z.object({ entities: z.array(z.string()), search: z.array(z.string()) });

class AdventureLLMPhaseMemoryFetch extends AdventureLLMPhase {

    constructor(llmClient: LLMClient, adventureState: AdventureState) {
        const config: AdventurePhaseConfig = {
            agentName: "Search Agent",
            prompts: [MEMORY_FETCH_PROMPT_PATH],
            prefill: "",
            saveMessageToHistory: true,
            retryCount: 3,
            llmParameters: {
                llmModel: OPENROUTER_MODEL_MEMORY_FETCH,
                maxTokens: 200,
                stopSequence: "",
                jsonOutput: true,
                schema: null,
                reasoning: null,
            },
        };
        super(llmClient, config, adventureState);
    }

    protected praparePhase() {
        this.llmClient.clearMessageHistory();
        this.llmClient.addMessage({
            role: "user",
            name: "Writer",
            content: this.adventureState.turns.at(-2)?.fullWriterResponse || "",
        });
        this.llmClient.addMessage({
            role: "user",
            name: "Player",
            content: yaml.dump(this.adventureState.getLastTurn().userInput, { lineWidth: -1 }),
        });
    }

    public async parsePhaseResult(): Promise<TurnValidationResult> {
        const result = new TurnValidationResult();
        const response = JSON.parse(cleanJSONResponse(this.accumulatedResponse)) as MemoryFetchResult;
        for (const entityId of response.entities) {
            if (this.adventureState.memoryGraph.entities[entityId.trim()]) {
                this.adventureState.addFetchedEntity(entityId.trim());
            } else {
                result.errors.push(`Invalid entity id '${entityId}'`);
            }
        }
        if (response.search.length === 0) {
            result.errors.push("No search terms provided");
        }
        const ENTITY_RESULT_COUNT = 5;
        let foundEntities = await this.adventureState.getEntitiesMemoryStore().searchMultiple(response.search, ENTITY_RESULT_COUNT, Object.keys(this.adventureState.fetchedEntities));
        for (const entity of foundEntities) {
            this.adventureState.addFetchedEntity(entity.chunkId);
        }
        const NARRATIVE_RESULT_COUNT = 10;
        let foundChunks = await this.adventureState.getNarrativeMemoryStore().searchMultiple(response.search, NARRATIVE_RESULT_COUNT);
        let searchResults = foundChunks.map((chunk) => {
            if (chunk.meta.paragraphId) {
                return `Turn ${chunk.meta.paragraphId[0]} p${chunk.meta.paragraphId[1]}: ${chunk.text}`;
            }
            return chunk.text;
        });

        if (result.isFailed()) {
            return result;
        }
        this.setParameterOverride("FETCHED_ENTITIES", this.adventureState.getFetchedEntities(false));
        this.setParameterOverride("SEARCHED_RESULTS", searchResults.join("\n"));
        return result;
    }

    public finalizeHistory() {
    }
}

class AdventureLLMPhaseAnalyze extends AdventureLLMPhase {

    constructor(llmClient: LLMClient, adventureState: AdventureState) {
        const config: AdventurePhaseConfig = {
            agentName: "Analyzer Agent",
            prompts: [ANALYZE_PROMPT_PATH],
            prefill: "",
            saveMessageToHistory: true,
            retryCount: 3,
            llmParameters: {
                llmModel: OPENROUTER_MODEL!,
                maxTokens: 1000,
                stopSequence: "</response>",
                jsonOutput: false,
                schema: null,
                reasoning: null,
            },
        };
        super(llmClient, config, adventureState);
    }


    protected praparePhase() {
    }

    public async parsePhaseResult(): Promise<TurnValidationResult> {
        const result = new TurnValidationResult();
        const response = findXMLSection(this.accumulatedResponse, "response", result);
        if (result.isFailed()) {
            return result;
        }
        findXMLSection(response!, "scene", result);
        findXMLSection(response!, "narrativePlan", result);
        if (result.isFailed()) {
            return result;
        }
        // this.adventureState.getLastTurn().fullAnalysis = response!;
        return result;
    }
}

class AdventureLLMPhaseNarrative extends AdventureLLMPhase {

    constructor(llmClient: LLMClient, adventureState: AdventureState) {
        const config: AdventurePhaseConfig = {
            agentName: "Writer Agent",
            prompts: [NARRATIVE_PROMPT_PATH],
            prefill: "",
            saveMessageToHistory: true,
            retryCount: 3,
            llmParameters: {
                llmModel: OPENROUTER_MODEL_NARRATIVE!,
                maxTokens: 4000,
                stopSequence: "</response>",
                jsonOutput: false,
                schema: null,
                reasoning: null,
            }
        };
        super(llmClient, config, adventureState);
    }

    protected praparePhase() {
        this.llmClient.clearMessageHistory();
        this.llmClient.addMessage({
            role: "user",
            name: "Deverloper",
            content: this.adventureState.resolvePrompt(HISTORY_PROMPT_PATH),
        });
        this.llmClient.addMessage({
            role: "user",
            name: "Deverloper",
            content: this.adventureState.resolvePrompt(MEMORY_FETCH_RESULT_PROMPT_PATH),
        });
        const firstHistoryTurn = this.adventureState.turns.length - 1 - TURNS_TO_KEEP_IN_HISTORY;
        const lastHistoryTurn = this.adventureState.turns.length - 2;
        for (const turn of this.adventureState.getRecentTurns(firstHistoryTurn, lastHistoryTurn)) {
            if (turn.userInput) {
                this.llmClient.addMessage({
                    role: "user",
                    name: "Player",
                    content: `## Turn ${turn.turnNumber} start\nPlayer input:\n${yaml.dump(turn.userInput, { lineWidth: -1 })}`,
                });
            }
            if (turn.turnNumber === 0) {
                this.llmClient.addMessage({
                    role: "user",
                    name: "Deverloper",
                    content: `Turn 0:\n${turn.fullWriterResponse}`,
                });
            }
            else {
                this.llmClient.addMessage({
                    role: "assistant",
                    name: "Writer Agent",
                    content: "<response>" + turn.fullWriterResponse + "</response>",
                });
            }
            if (turn.criticFeedback || turn.feedback) {
                let feedbackMessage = `## Turn ${turn.turnNumber} end\n`;
                if (turn.criticFeedback) {
                    feedbackMessage += `Critic feedback: ${turn.criticFeedback}\n`;
                }
                if (turn.feedback) {
                    feedbackMessage += `Player feedback: ${yaml.dump(turn.feedback, { lineWidth: -1 })}`;
                }
                this.llmClient.addMessage({
                    role: "user",
                    name: "Developer",
                    content: feedbackMessage,
                });
            }
        }
        this.llmClient.addMessage({
            role: "user",
            name: "Player",
            content: yaml.dump(this.adventureState.getLastTurn().userInput, { lineWidth: -1 }),
        });
    }

    public async parsePhaseResult(): Promise<TurnValidationResult> {
        const result = new TurnValidationResult();
        let response = findXMLSection(this.accumulatedResponse, "response", new TurnValidationResult());
        const sceneSection = findXMLSection(response || this.accumulatedResponse, "scene", result);
        const narrativeSection = findXMLSection(response || this.accumulatedResponse, "narrative", result);
        const notesSection = findXMLSection(response || this.accumulatedResponse, "notes", result);
        const suggestedActionsSection = findXMLSection(response || this.accumulatedResponse, "suggestedActions", result);
        if (result.isFailed()) {
            return result;
        }
        if (!response) {
            // Workaround for case when LLM forgot the <response> tag but wrote otherwise correct response
            this.accumulatedResponse = "<response>" + this.accumulatedResponse + "</response>";
            response = findXMLSection(this.accumulatedResponse, "response", result);
        }
        if (result.isFailed()) {
            return result;
        }
        this.adventureState.getLastTurn().fullWriterResponse = response!.trim();
        this.adventureState.getLastTurn().suggestedActions = suggestedActionsSection!.trim();
        return result;
    }

    protected addTextChunk(chunk: string, onTurnUpdated: () => void) {
        this.accumulatedResponse += chunk;
        this.adventureState.getLastTurn().fullWriterResponse = findResponsePartialSection(this.accumulatedResponse, "response") || "";
        if (getTurnNarrative(this.adventureState.getLastTurn(), true).length > 0) {
            onTurnUpdated();
        }
    }

    protected findResponsePartialSection(response: string, sectionName: string): string | null {
        const fullSection = findXMLSection(response, sectionName, new TurnValidationResult());
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
    feedback: string;
    newEntities: MemoryGraphUpdate;
    updates: MemoryGraphUpdate;
    backgroundPrompt: string;
    illustrationType: string;
    illustrationPrompt: string;
    playerPortraitPrompt: string;
}


class AdventureLLMPhaseMemoryUpdate extends AdventureLLMPhase {

    constructor(llmClient: LLMClient, adventureState: AdventureState) {
        const config: AdventurePhaseConfig = {
            agentName: "Assistant Agent",
            prompts: [ASSISTANT_PROMPT_PATH],
            prefill: "",
            saveMessageToHistory: false,
            retryCount: 5,
            llmParameters: {
                llmModel: OPENROUTER_MODEL_ASSISTANT!,
                maxTokens: 3000,
                stopSequence: "",
                jsonOutput: true,
                schema: null,
                reasoning: null,
            }
        };
        super(llmClient, config, adventureState);
    }

    protected praparePhase() {
        this.llmClient.clearMessageHistory();
        const savedRecentTurns = this.adventureState.parameters["RECENT_TURNS"];
        this.adventureState.parameters["RECENT_TURNS"] = " ";
        this.llmClient.addMessage({
            role: "user",
            name: "Deverloper",
            content: this.adventureState.resolvePrompt(HISTORY_PROMPT_PATH),
        });
        this.adventureState.parameters["RECENT_TURNS"] = savedRecentTurns;

        
        const turn = this.adventureState.getLastTurn();
        this.llmClient.addMessage({
            role: "user",
            name: "Player",
            content: `## Turn ${turn.turnNumber} start\nPlayer input:\n${yaml.dump(turn.userInput, { lineWidth: -1 })}`,
        });
        this.llmClient.addMessage({
            role: "user",
            name: "Writer Agent",
            content: turn.fullWriterResponse,
        });
    }

    public async parsePhaseResult(): Promise<TurnValidationResult> {
        const result = new TurnValidationResult();
        const response = JSON.parse(cleanJSONResponse(this.accumulatedResponse)) as LLMResponseMemoryUpdate;
        if (response.feedback) {
            const narrativeWordCount = getTurnNarrative(this.adventureState.getLastTurn(), false).split(/\s+/).length;
            let criticFeedback = `Feedback: Narrative word count ${narrativeWordCount}`;
            if (narrativeWordCount < 500) {
                criticFeedback += ` CRITICAL: Narrative didn't reach the minimum word count. Next turn should overcompensate for this.\n`;
            } else if (narrativeWordCount < 600) {
                criticFeedback += ` Narrative length is dangerously low. Try writing more next time.\n`;
            } else {
                criticFeedback += ` Narrative length is good. Keep it up!\n`;
            }
            this.adventureState.getLastTurn().criticFeedback = criticFeedback + response.feedback;
        }
        if (response.newEntities) {
            const existingNewEntities = Object.keys(response.newEntities).filter((key) => this.adventureState.memoryGraph.entities[key]);
            if (existingNewEntities.length > 0) {
                result.errors.push(`'newEntities' section contains entities already present in memory: ${existingNewEntities.join(", ")}`);
            }
        }
        if (response.updates) {
            for (const entityId of Object.keys(response.updates)) {
                const entityUpdate = response.updates[entityId];
                const entity = this.adventureState.memoryGraph.entities[entityId];
                if (!entity) {
                    result.errors.push(`Entity ${entityId} must be added to memory before updating.`);
                }
                else if (entityUpdate.info) {
                    const combinedInfo = entity.info + "\n" + entityUpdate.info;
                    entityUpdate.info = combinedInfo;
                } else if (entityUpdate.secret) {
                    const combinedSecret = entity.secret + "\n" + entityUpdate.secret;
                    entityUpdate.secret = combinedSecret;
                }
            }
        }
        if (!response.illustrationType) {
            result.errors.push(`illustrationType is missing`);
        }
        if (!response.backgroundPrompt) {
            result.errors.push("backgroundPrompt is missing");
        }
        if (!response.illustrationPrompt) {
            result.errors.push("illustrationPrompt is missing");
        }
        if (!response.playerPortraitPrompt) {
            result.errors.push("playerPortraitPrompt is missing");
        }
        if (result.isFailed()) {
            return result;
        }
        if (response.newEntities) {
            result.errors.push(...this.updateMemory(response.newEntities).errors);
        }
        if (response.updates) {
            result.errors.push(...this.updateMemory(response.updates).errors);
        }
        if (result.isFailed()) {
            return result;
        }
        this.adventureState.getLastTurn().illustrationType = response.illustrationType;
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
            const illustrationPrompt = this.adventureState.makeImageUpdate("illustration", response.illustrationPrompt + ", located in " + settingPrompt, response.illustrationType);
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
            agentName: "Summary Agent",
            prompts: [SUMMARY_PROMPT_PATH],
            prefill: "",
            saveMessageToHistory: false,
            retryCount: 5,
            llmParameters: {
                llmModel: OPENROUTER_MODEL!,
                maxTokens: 2000,
                stopSequence: "",
                jsonOutput: true,
                schema: null,
                reasoning: null,
            }
        };
        super(llmClient, config, adventureState);
    }

    protected praparePhase() {
        this.llmClient.addMessage({
            role: "user",
            name: "Developer",
            content: `Summary phase`,
        });
    }

    public async parsePhaseResult(): Promise<TurnValidationResult> {
        const result = new TurnValidationResult();
        const response = JSON.parse(cleanJSONResponse(this.accumulatedResponse)) as LLMResponseSummary;
        if (!response.summary) {
            result.errors.push("Summary is missing");
        }
        if (!response.plotPlan) {
            result.errors.push("Plot plan is missing");
        }
        if (!response.userProfile) {
            result.errors.push("User profile is missing");
        }
        if (result.isFailed()) {
            return result;
        }
        this.adventureState.parameters["PLOT_PLAN"] = response.plotPlan;
        this.adventureState.parameters["USER_PROFILE"] = response.userProfile;
        this.adventureState.parameters["SUMMARY_ANALYSIS"] = response.analysis;
        logger.info("Summary:", response.summary);
        const summary = this.adventureState.getParameterOrDefault("STORY_ARCHIVE", "") + response.summary;
        this.adventureState.parameters["STORY_ARCHIVE"] = summary;
        this.adventureState.lastSummarizedTurn = this.adventureState.turns.length - 1;
        return result;
    }
}


class AdventureLLMPhaseCritic extends AdventureLLMPhase {

    constructor(llmClient: LLMClient, adventureState: AdventureState) {
        const config: AdventurePhaseConfig = {
            agentName: "Critic Agent",
            prompts: [CRITIC_PROMPT_PATH],
            prefill: "",
            saveMessageToHistory: true,
            retryCount: 2,
            llmParameters: {
                llmModel: OPENROUTER_MODEL!,
                maxTokens: 1500,
                stopSequence: "",
                jsonOutput: false,
                schema: null,
                reasoning: null,
            }
        };
        super(llmClient, config, adventureState);
    }

    protected praparePhase() {
        this.llmClient.addMessage({
            role: "user",
            name: "Developer",
            content: `## Write the short feedback on the current turn`,
        });
    }

    public async parsePhaseResult(): Promise<TurnValidationResult> {
        const result = new TurnValidationResult();
        const response = findXMLSection(this.accumulatedResponse, "response", result);
        if (result.isFailed()) {
            return result;
        }
        this.adventureState.getLastTurn().criticFeedback = response!;
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
        return this.adventureState.getRecentTurns(firstTurn, lastTurnInclusive);
    }

    private fetchRecentTurnNarratives(firstTurn: number, lastTurnInclusive: number): string {
        let recentTurns = this.getRecentTurns(firstTurn, lastTurnInclusive);
        let recentTurnsText = "";
        for (const turn of recentTurns) {
            recentTurnsText += `\n--Turn ${turn.turnNumber}:\n`;
            if (turn.userInput) {
                recentTurnsText += `>${yaml.dump(turn.userInput, { lineWidth: -1 })}\n`;
            }
            recentTurnsText += getTurnNarrative(turn, false);
            recentTurnsText += "\nNotes:\n"
            recentTurnsText += findXMLSection(turn.fullWriterResponse, "notes", new TurnValidationResult()) || "";
            if (turn.feedback) {
                recentTurnsText += `Player feedback: ${yaml.dump(turn.feedback, { lineWidth: -1 })}\n`;
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

        for (let i = 0; i < firstUnarchivedTurn; i++) {
            if (!this.adventureState.getNarrativeMemoryStore().isTurnKnown(i)) {
                let narrative = getTurnNarrative(this.adventureState.turns[i], false);
                this.adventureState.getNarrativeMemoryStore().upsertNarrative(i, narrative);
            }
        }

        const turnNumber = this.adventureState.turns.length;
        this.adventureState.turns.push({
            turnNumber: turnNumber,
            fullWriterResponse: "",
            suggestedActions: "",
            images: [],
            userInput: userInput,
            illustrationType: "",
        });
        this.eventEmitter.emit("turn-updated", this.adventureState.getLastTurn());
        this.adventureState.parameters["RECENT_TURNS"] = this.fetchRecentTurnNarratives(firstUnarchivedTurn, firstHistoryTurn - 1);
        this.adventureState.parameters["TURN_NUMBER"] = turnNumber.toString();
        this.adventureState.parameters["REFMAP"] = Object.values(this.adventureState.memoryGraph.entities).map((e) => `${e.id} → ${e.name}${e.brief ? ", " + e.brief : ""}`).join("\n");
        this.adventureState.parameters["EXISTING_ENTITY_IDS"] = Object.keys(this.adventureState.memoryGraph.entities).join(",");
        this.adventureState.parameters["SEARCHED_RESULTS"] = "";
        this.updateFetchedEntities(false); // TODO: fetch all entities if the turn is a new adventure
        // if (!await this.runPhase(AdventurePhase.Init, {})) {
        //     return;
        // }
        // this.llmClient.clearMessageHistory();

        let phases: AdventureLLMPhase[] = [];
        phases.push(...[
            new AdventureLLMPhaseMemoryFetch(this.llmClient, this.adventureState),
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
            await this.runPhase(new AdventureLLMPhaseSummary(this.llmClient, this.adventureState));
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

    private updateFetchedEntities(fetchAll: boolean) {
        this.adventureState.parameters["FETCHED_ENTITIES"] = this.adventureState.getFetchedEntities(fetchAll);
    }
}
