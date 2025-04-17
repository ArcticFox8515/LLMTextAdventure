import { EventEmitter } from "stream";
import { logger } from "./logger";
import fs from 'fs';
import * as yaml from 'js-yaml';
import { LLMClient } from "./llm-client";
import { AdventureImageUpdate, AdventureState, AdventureTurnFeedback, AdventureTurnInfo, AdventureUserInput, ImagePromptParameters, StoryStartingParameters } from "./adventure-types";
import { AdventureLLMRequest } from "./adventure-llm-request";

export function loadStoryParameters(filePath: string): StoryStartingParameters | null {
    try {
        const fileContents = fs.readFileSync(filePath, 'utf8');
        const data = yaml.load(fileContents) as StoryStartingParameters;
        return data;
    } catch (error) {
        console.error('Error loading YAML file:', error);
        return null;
    }
}

export class Adventure {
    // Transient fields
    private llmClient: LLMClient;
    private llmRunning: boolean = false;
    private backgroundImagePreChanged = false;
    private static readonly MAX_SAVE_FILES = 4; // Number of save files to keep

    private eventEmitter: EventEmitter = new EventEmitter();
    private runningRequest: AdventureLLMRequest | null = null;

    // Saved fields
    private state: AdventureState;

    constructor(llmClient: LLMClient) {
        this.llmClient = llmClient;
        this.state = new AdventureState();
    }

    public async start() {
        if (fs.existsSync("saved/adventure-state.yaml")) {
            await this.state.deserialize(fs.readFileSync("saved/adventure-state.yaml", "utf-8"));
        }
    }

    public onLLMStartStop(callback: (isRunning: boolean) => void) {
        this.eventEmitter.on("llm-start-stop", callback);
    }

    public onMessageChanged(callback: (turnMessage: AdventureTurnInfo) => void) {
        this.eventEmitter.on("message-changed", callback);
    }

    public onImageChanged(callback: (update: AdventureImageUpdate) => void) {
        this.eventEmitter.on("image-changed", callback);
    }

    public isLLMRunning(): boolean {
        return this.llmRunning;
    }

    public getLastTurn(): AdventureTurnInfo {
        return this.state.turns[this.state.turns.length - 1];
    }

    public getImageParameters(): ImagePromptParameters {
        return this.state.imagePromptParameters;
    }

    public getTurnNumber(): number {
        return this.getLastTurn().turnNumber;
    }

    public getAllTurns(): AdventureTurnInfo[] {
        return this.state.turns;
    }

    public async startAdventure(parameters: StoryStartingParameters) {
        logger.info("\n\n*** Starting adventure ***\n\n");
        this.state = new AdventureState();
        await this.state.initAdventure(parameters);

        this.emitLastTurnUpdate();
        await this.performTurn("", "[first action from automated system]: " + this.state.getParameterOrDefault("FIRST_INPUT", "Begin the story"));
    }

    private createUserMessage(characterAction?: string, instructions?: string): AdventureUserInput {
        let query: AdventureUserInput = {
            action: characterAction,
            outOfCharacter: instructions || undefined,
        };
        return query;
    }

    public async performTurn(characterAction?: string, instructions?: string) {
        this.setLLMRunning(true);

        const turnRequest = new AdventureLLMRequest(this.llmClient, this.state);
        turnRequest.onTurnUpdated((turnInfo) => {
            this.emitLastTurnUpdate();
        });
        const result = await turnRequest.performTurn(this.createUserMessage(characterAction, instructions));
        this.setLLMRunning(false);
        if (result.errors.length > 0) {
            this.rollbackTurn();
            return;
        }
        this.saveState();
        this.emitLastTurnUpdate();
        this.getLastTurn().images.forEach(image => {
            this.eventEmitter.emit("image-changed", image);
        });
    }

    public addFeedback(feedback: AdventureTurnFeedback) {
        this.state.getLastTurn().feedback = feedback;
    }

    private setLLMRunning(running: boolean) {
        this.llmRunning = running;
        this.eventEmitter.emit("llm-start-stop", running);
    }

    private rollbackTurn() {
        this.eventEmitter.emit("message-changed", { turnNumber: this.getTurnNumber() + 1, narrative: "<TURN CANCELLED>", suggestedActions: "" });
    }

    public isAdventureStarted(): boolean {
        return this.state.turns.length > 1;
    }

    private emitLastTurnUpdate() {
        this.eventEmitter.emit("message-changed", this.state.turns[this.state.turns.length - 1]);
    }

    public getVisibleTurn(): AdventureTurnInfo {
        return this.state.turns.length > 0 ? this.state.turns[this.state.turns.length - 1] : { turnNumber: 0, narrativePlan: "", narrative: "", suggestedActions: "", images: [], analysis: "", scene: "", illustrationId: "" };
    }

    private saveState() {
        const basePath = "saved";
        const baseFilename = "adventure-state";
        const extension = ".yaml";
        const mainFile = `${basePath}/${baseFilename}${extension}`;

        // Check if the directory exists and create it if not
        if (!fs.existsSync(basePath)) {
            fs.mkdirSync(basePath, { recursive: true });
        }

        // Check for the oldest backup (MAX_SAVE_FILES) and delete if it exists
        const oldestBackupPath = `${basePath}/${baseFilename}-${Adventure.MAX_SAVE_FILES}${extension}`;
        if (fs.existsSync(oldestBackupPath)) {
            fs.unlinkSync(oldestBackupPath);
        }

        // Shift existing backups: rename i.yaml to (i+1).yaml starting from the highest
        for (let i = Adventure.MAX_SAVE_FILES - 1; i > 0; i--) {
            const currentBackup = `${basePath}/${baseFilename}-${i}${extension}`;
            const nextBackup = `${basePath}/${baseFilename}-${i + 1}${extension}`;

            if (fs.existsSync(currentBackup)) {
                fs.renameSync(currentBackup, nextBackup);
            }
        }

        // Rename the current save file to adventure-state-1.yaml if it exists
        if (fs.existsSync(mainFile)) {
            const firstBackup = `${basePath}/${baseFilename}-1${extension}`;
            fs.renameSync(mainFile, firstBackup);
        }

        // Save the current state as the main file
        fs.writeFileSync(mainFile, this.state.serialize(), 'utf8');
    }
}
