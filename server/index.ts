import express from "express";
import bodyParser from "body-parser";
import cors from "cors";
import dotenv from "dotenv";
import path from "path";
import fs from "fs";
import { WebSocket, WebSocketServer } from "ws";
import { LLMClient } from "./llm-client";
import { MCPClient } from "./mcp-client";
import { logger } from "./logger";
import { AdventureImageUpdate, AdventureTurnInfo, ImageRole } from "./adventure-types";
import { Adventure, loadStoryParameters } from "./adventure";
import { GeneratorOptions, ImageGenerator } from "./image-provider";

dotenv.config();

const app = express();
const PORT = process.env.PORT || 3001;

// Middleware
app.use(cors());
app.use(bodyParser.json());

// Serve static files from the React app
app.use(express.static(path.join(__dirname, '../../dist/client')));

// Get environment variables
const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY;
if (!OPENROUTER_API_KEY) {
  throw new Error("OPENROUTER_API_KEY is not set");
}
const OPENROUTER_API_URL = process.env.OPENROUTER_API_URL || "https://openrouter.ai/api/v1";
const STORY_PARAMETERS_PATH = "prompts/story/story-parameters.yaml";

if (!fs.existsSync("saved")) {
  fs.mkdirSync("saved");
}

class ClientSession {
  // Create application state
  private mcpClient = new MCPClient();
  private llmClient = new LLMClient(OPENROUTER_API_KEY!, OPENROUTER_API_URL, this.mcpClient);
  private adventure = new Adventure(this.llmClient);
  private imageGenerator = new ImageGenerator();
  private ws: WebSocket | null = null;

  public async start() {
    await this.mcpClient.start();
    this.adventure.onLLMStartStop((isRunning) => {
      this.ws?.send(JSON.stringify({ type: "llm-running", content: isRunning }));
    });
    this.adventure.onMessageChanged((turnInfo) => {
      this.sendTurnUpdate(turnInfo);
    });
    this.adventure.onImageChanged((imageUpdate) => {
      this.sendImageUpdate(imageUpdate);
    });
    await this.adventure.start();
  }

  private async startNewAdventure() {
    const parameters = loadStoryParameters(STORY_PARAMETERS_PATH);
    if (parameters) {
      await this.adventure.startAdventure(parameters);
    }
  }

  private sendTurnUpdate(turnInfo: AdventureTurnInfo) {
    this.ws?.send(JSON.stringify({ type: "turn-update", content: turnInfo }));
  }

  private sendImageUpdate(imageUpdate: AdventureImageUpdate) {
    logger.info("Image update: ", imageUpdate.role, " = ", imageUpdate.imagePrompt);
    if (imageUpdate.imagePrompt === "") {
      return;
    }
    let generatorOptions: GeneratorOptions = {
      model: this.adventure.getImageParameters().model,
      prompt: imageUpdate.imagePrompt,
      negativePrompt: imageUpdate.negativePrompt
    };
    this.imageGenerator.generateImage(generatorOptions).then((images) => {
      if (images.length > 0) {
        const imageData = images.map((image) => image.toString("base64"));
        this.ws?.send(JSON.stringify({ type: "image-update", role: imageUpdate.role, content: imageData }));
      }
    });
  }

  private async onActionMessage(message: any) {
    try {
      const { characterAction, instructions } = message;
      if (!this.adventure.isAdventureStarted()) {
        await this.startNewAdventure();
      } else {
        this.adventure.performTurn(characterAction, instructions);
      }
    } catch (error) {
      console.error("Error processing query:", error);
      this.ws?.send(JSON.stringify({ type: "error", error: "Failed to process query" }));
    }
  }

  private async onRefreshImageMessage(message: any) {
    const imageUpdate = this.adventure.getLastTurn().images.find((image) => {
      return image.role === message.role;
    });
    if (!imageUpdate) {
      console.error("Image not found for role:", message.role);
      console.error(JSON.stringify(this.adventure.getLastTurn().images));
      return;
    }
    let generatorOptions: GeneratorOptions = {
      model: this.adventure.getImageParameters().model,
      prompt: imageUpdate.imagePrompt,
      negativePrompt: imageUpdate.negativePrompt
    };
    this.imageGenerator.clearImage(generatorOptions);
    this.sendImageUpdate(imageUpdate);
  }

  private async onFeedbackMessage(message: any) {
    try {
      const { feedbackType, feedbackComment } = message;
      logger.info(`Feedback received: ${feedbackType} - ${feedbackComment}`);
      this.adventure.addFeedback({
        feedbackType: feedbackType,
        feedbackComment: feedbackComment,
      });
    } catch (error) {
      console.error("Error processing feedback:", error);
      this.ws?.send(JSON.stringify({ type: "error", error: "Failed to process feedback" }));
    }
  }

  setWebSocket(ws: WebSocket | null) {
    this.ws = ws;
    if (ws === null) {
      return;
    }
    this.adventure.getAllTurns().map((turnInfo) => {
      this.sendTurnUpdate(turnInfo);
    });
    if (this.adventure.getLastTurn()) {
      this.adventure.getLastTurn().images.map((imageUpdate) => {
        this.sendImageUpdate(imageUpdate);
      });
    }

    ws.on("message", async (message: string) => {
      const parsedMessage = JSON.parse(message);
      if (parsedMessage.type === "action") {
        await this.onActionMessage(parsedMessage);
      } else if (parsedMessage.type === "refresh-image") {
        await this.onRefreshImageMessage(parsedMessage);
      } else if (parsedMessage.type === "feedback") {
        await this.onFeedbackMessage(parsedMessage);
      } else {
        console.error("Unknown message type:", parsedMessage.type);
      }
    });
  }
}

const globalSession = new ClientSession();

// Create WebSocket server
const wss = new WebSocketServer({ port: 3002 });
wss.on("connection", (ws: WebSocket) => {
  logger.info("WebSocket connection established");

  globalSession.setWebSocket(ws);

  ws.on("close", () => {
    logger.info("WebSocket connection closed");
    globalSession.setWebSocket(null);
  });
});

// API Routes can go here
// app.get('/api/something', (req, res) => { ... });

// For any request that doesn't match an API route, send the React app
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, '../../dist/client/index.html'));
});

// Start the server
(async () => {
  await globalSession.start();

  logger.info(JSON.stringify(loadStoryParameters(STORY_PARAMETERS_PATH)));

  app.listen(PORT, () => {
    logger.info(`Server running on http://localhost:${PORT}`);
    logger.info(`WebSocket server running on ws://localhost:3002`);
  });
})();