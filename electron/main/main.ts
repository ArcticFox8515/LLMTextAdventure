import { app, BrowserWindow, ipcMain } from 'electron';
import * as path from 'path';
import * as isDev from 'electron-is-dev';
import { spawn, ChildProcess } from 'child_process';
import { WebSocket, WebSocketServer } from 'ws';
import { LLMClient } from '../../server/llm-client';
import { MCPClient } from '../../server/mcp-client';
import { Adventure, loadStoryParameters } from '../../server/adventure';
import { ImageGenerator } from '../../server/image-provider';
import { logger } from '../../server/logger';
import { AdventureImageUpdate, AdventureTurnInfo } from '../../server/adventure-types';
import { getTurnNarrative } from '../../server/adventure-llm-request';
import fs from 'fs';
import dotenv from 'dotenv';

dotenv.config();

class ElectronApp {
  private mainWindow: BrowserWindow | null = null;
  private backendProcess: ChildProcess | null = null;
  private wss: WebSocketServer | null = null;
  private mcpClient: MCPClient = new MCPClient();
  private llmClient: LLMClient | null = null;
  private adventure: Adventure | null = null;
  private imageGenerator: ImageGenerator = new ImageGenerator();

  constructor() {
    this.initializeApp();
  }

  private async initializeApp(): Promise<void> {
    await app.whenReady();
    await this.startBackend();
    this.createWindow();
    this.setupAppEventHandlers();
  }

  private async startBackend(): Promise<void> {
    const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY;
    const OPENROUTER_API_URL = process.env.OPENROUTER_API_URL || "https://openrouter.ai/api/v1";
    const STORY_PARAMETERS_PATH = "prompts/story/story-parameters.yaml";

    if (!OPENROUTER_API_KEY) {
      throw new Error("OPENROUTER_API_KEY is not set");
    }

    if (!fs.existsSync("saved")) {
      fs.mkdirSync("saved");
    }

    // Initialize backend components
    await this.mcpClient.start();
    this.llmClient = new LLMClient(OPENROUTER_API_KEY, OPENROUTER_API_URL, this.mcpClient);
    this.adventure = new Adventure(this.llmClient);

    // Set up adventure event handlers
    this.adventure.onLLMStartStop((isRunning) => {
      this.sendToRenderer('llm-running', isRunning);
    });

    this.adventure.onMessageChanged((turnInfo) => {
      this.sendTurnUpdate(turnInfo);
    });

    this.adventure.onImageChanged((imageUpdate) => {
      this.sendImageUpdate(imageUpdate);
    });

    await this.adventure.start();

    // Create WebSocket server for frontend communication
    this.wss = new WebSocketServer({ port: 3002 });
    this.wss.on("connection", (ws: WebSocket) => {
      logger.info("WebSocket connection established");

      // Send existing turns and images when client connects
      if (this.adventure) {
        this.adventure.getAllTurns().forEach((turnInfo) => {
          this.sendTurnUpdate(turnInfo);
        });
        
        const lastTurn = this.adventure.getLastTurn();
        if (lastTurn) {
          lastTurn.images.forEach((imageUpdate) => {
            this.sendImageUpdate(imageUpdate);
          });
        }
      }

      ws.on("message", async (message: string) => {
        const parsedMessage = JSON.parse(message);
        await this.handleWebSocketMessage(parsedMessage);
      });

      ws.on("close", () => {
        logger.info("WebSocket connection closed");
      });
    });

    logger.info("Backend started successfully");
  }

  private async handleWebSocketMessage(message: any): Promise<void> {
    if (!this.adventure) return;

    try {
      if (message.type === "action") {
        const { characterAction, instructions } = message;
        if (!this.adventure.isAdventureStarted()) {
          await this.startNewAdventure();
        } else {
          this.adventure.performTurn(characterAction, instructions);
        }
      } else if (message.type === "refresh-image") {
        await this.onRefreshImageMessage(message);
      } else if (message.type === "feedback") {
        const { feedbackType, feedbackComment } = message;
        logger.info(`Feedback received: ${feedbackType} - ${feedbackComment}`);
        this.adventure.addFeedback({
          feedbackType: feedbackType,
          feedbackComment: feedbackComment,
        });
      }
    } catch (error) {
      console.error("Error processing WebSocket message:", error);
    }
  }

  private async startNewAdventure(): Promise<void> {
    if (!this.adventure) return;
    
    const STORY_PARAMETERS_PATH = "prompts/story/story-parameters.yaml";
    const parameters = loadStoryParameters(STORY_PARAMETERS_PATH);
    if (parameters) {
      await this.adventure.startAdventure(parameters);
    }
  }

  private async onRefreshImageMessage(message: any): Promise<void> {
    if (!this.adventure) return;

    const imageUpdate = this.adventure.getLastTurn().images.find((image) => {
      return image.role === message.role;
    });
    
    if (!imageUpdate) {
      console.error("Image not found for role:", message.role);
      return;
    }

    // Regenerate image
    const generatorOptions = {
      model: this.adventure.getImageParameters().model,
      prompt: imageUpdate.imagePrompt,
      negativePrompt: imageUpdate.negativePrompt
    };
    
    this.imageGenerator.clearImage(generatorOptions);
    this.sendImageUpdate(imageUpdate);
  }

  private sendTurnUpdate(turnInfo: AdventureTurnInfo): void {
    const turnViewModel = {
      turnNumber: turnInfo.turnNumber,
      narrative: this.getTurnNarrative(turnInfo, true),
      suggestedActions: turnInfo.suggestedActions,
      userInput: turnInfo.userInput,
    };
    this.broadcastToWebSocket({ type: "turn-update", content: turnViewModel });
  }

  private sendImageUpdate(imageUpdate: AdventureImageUpdate): void {
    if (!this.adventure) return;

    logger.info("Image update:", imageUpdate.role, "=", imageUpdate.imagePrompt);
    if (imageUpdate.imagePrompt === "") {
      return;
    }

    const generatorOptions = {
      model: this.adventure.getImageParameters().model,
      prompt: imageUpdate.imagePrompt,
      negativePrompt: imageUpdate.negativePrompt
    };

    this.imageGenerator.generateImage(generatorOptions).then((images) => {
      if (images.length > 0) {
        const imageData = images.map((image) => image.toString("base64"));
        this.broadcastToWebSocket({ 
          type: "image-update", 
          role: imageUpdate.role, 
          content: imageData 
        });
      }
    });
  }
  private getTurnNarrative(turnInfo: AdventureTurnInfo, includeUserAction: boolean): string {
    // Use the imported getTurnNarrative function
    return getTurnNarrative(turnInfo, includeUserAction);
  }

  private broadcastToWebSocket(data: any): void {
    if (this.wss) {
      this.wss.clients.forEach((client) => {
        if (client.readyState === WebSocket.OPEN) {
          client.send(JSON.stringify(data));
        }
      });
    }
  }

  private sendToRenderer(channel: string, data: any): void {
    if (this.mainWindow && !this.mainWindow.isDestroyed()) {
      this.mainWindow.webContents.send(channel, data);
    }
  }

  private createWindow(): void {
    this.mainWindow = new BrowserWindow({
      height: 800,
      width: 1200,      webPreferences: {
        nodeIntegration: false,
        contextIsolation: true,
        preload: path.join(__dirname, '../preload/preload.js'),
      },
    });    const url = false // Force production mode 
      ? 'http://localhost:3000' 
      : `file://${path.join(__dirname, '../../../renderer/index.html')}`;
    
    this.mainWindow.loadURL(url);    if (false) { // Don't open DevTools in production
      this.mainWindow?.webContents.openDevTools();
    }

    this.mainWindow.on('closed', () => {
      this.mainWindow = null;
    });
  }

  private setupAppEventHandlers(): void {
    app.on('window-all-closed', () => {
      if (process.platform !== 'darwin') {
        this.cleanup();
        app.quit();
      }
    });

    app.on('activate', () => {
      if (BrowserWindow.getAllWindows().length === 0) {
        this.createWindow();
      }
    });

    app.on('before-quit', () => {
      this.cleanup();
    });
  }

  private cleanup(): void {
    if (this.backendProcess) {
      this.backendProcess.kill();
      this.backendProcess = null;
    }
    
    if (this.wss) {
      this.wss.close();
      this.wss = null;
    }
  }
}

// Initialize the Electron app
new ElectronApp();
