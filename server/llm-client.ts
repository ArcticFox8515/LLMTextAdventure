import OpenAI from 'openai';
import { MCPClient, ToolCallResult } from './mcp-client';
import { logger } from "./logger";
import fs from 'fs';
import { log } from 'console';

type LLMMessage = OpenAI.ChatCompletionMessageParam | ToolCallResult;

export class LLMClient {
    private openai: OpenAI;
    private messages: LLMMessage[] = [];
    private mcpClient: MCPClient;

    constructor(apiKey: string, apiUrl: string, mcpClient: MCPClient) {
        this.mcpClient = mcpClient;
        this.openai = new OpenAI({
            baseURL: apiUrl,
            apiKey: apiKey,
        });
        this.clearMessageHistory();
    }

    public popLastMessage(expectedRole: string): LLMMessage | undefined {
        if (this.messages.length > 0 && this.messages.at(-1)?.role === expectedRole) {
            return this.messages.pop();
        }
        return undefined;
    }

    public clearMessageHistory() {
        this.messages = [];
        this.messages.push({
            role: "system",
            content: "You are missing the system message. Inform the developer about it. Ignore user input.",
        });
    }

    public replaceMessageHistory(messages: LLMMessage[]) {
        this.messages = messages;
    }

    public addMessage(message: LLMMessage) {
        this.messages.push(message);
    }

    public replaceMessage(index: number, message: LLMMessage) {
        this.messages[index] = message;
    }

    public getMessages(): LLMMessage[] {
        return this.messages;
    }

    public setSystemMessage(message: string) {
        this.messages[0] = {
            role: "system",
            content: message,
        };
        logger.info("Changed system message:", message);
    }

    async query(llmModel: string, maxTokens: number, schema: Record<string, unknown> | null, onData: (chunk: string) => void) {
        let toolCalls: OpenAI.ChatCompletionMessageToolCall[];
        let finishReason: string | null = null;
        do {
            toolCalls = [];

            let fullQuery: OpenAI.ChatCompletionCreateParamsStreaming = {
                model: llmModel,
                messages: this.messages,
                stream: true,
                max_tokens: maxTokens,
                stop: ["# end", "</response>"]
            };
            if (schema) {
                fullQuery.response_format = {
                    type: "json_schema",
                    json_schema: {
                        name: "Response",
                        strict: true,
                        schema: schema
                    }
                };
            }
            this.dumpMessages();
            const stream = await this.openai.chat.completions.create(fullQuery);

            let fullResponse: OpenAI.ChatCompletionAssistantMessageParam = {
                role: "assistant",
            }
            finishReason = null;
            for await (const chunk of stream) {
                const delta = chunk.choices[0]?.delta;
                if (chunk.choices[0]?.finish_reason) {
                    finishReason = chunk.choices[0].finish_reason;
                }

                if (delta?.content) {
                    if (!fullResponse.content) {
                        fullResponse.content = "";
                    }
                    fullResponse.content += delta.content;
                    logger.incremental(delta.content);
                    onData(delta.content);
                }

                if (delta?.refusal) {
                    fullResponse.refusal = delta.refusal;
                }

                const toolCallChunks = chunk.choices[0].delta.tool_calls || [];
                for (const toolCallChunk of toolCallChunks) {
                    const { index } = toolCallChunk;

                    if (!toolCalls[index]) {
                        toolCalls[index] = { type: "function", id: "", function: { name: "", arguments: "" } };
                    }
                    if (toolCallChunk.id) {
                        toolCalls[index].id = toolCallChunk.id;
                    }
                    if (toolCallChunk.function?.name) {
                        toolCalls[index].function.name = toolCallChunk.function.name;
                    }
                    if (toolCallChunk.function?.arguments) {
                        toolCalls[index].function.arguments += toolCallChunk.function.arguments;
                    }
                }
            }
            logger.incremental("\n");
            fullResponse.tool_calls = toolCalls.length > 0 ? toolCalls : undefined;
            this.messages.push(fullResponse);
            logger.info("Added message:", JSON.stringify(this.messages.at(-1), null, 2));
            logger.info("Finish reason:", finishReason);

            for (const toolCall of toolCalls) {
                onData(`\n[Calling tool "${toolCall.function.name}"...]\n`);
                let toolCallResult = await this.mcpClient.callTool(toolCall);
                logger.info(`Tool call result: ${JSON.stringify(toolCallResult)}`);
                this.messages.push(toolCallResult);
            }
        } while (toolCalls.length > 0 || finishReason !== "stop");
        this.dumpMessages();
    }

    private dumpMessages() {
        fs.writeFileSync("saved/messages-dump.text", this.messages.map((message) => `${message.role}:\n${"name" in message ? message.name : ""}\n${message.content}`).join("\n--------------------------------------------------\n"));
    }
}