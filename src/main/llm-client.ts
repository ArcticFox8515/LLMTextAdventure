import OpenAI from 'openai';
import { MCPClient, ToolCallResult } from './mcp-client';
import { logger } from "./logger";
import fs from 'fs';
import { zodToJsonSchema } from 'zod-to-json-schema';
import { LLMCallParameters } from './adventure-types';

// Extended interface for OpenAI parameters with provider support
interface ExtendedChatCompletionParams extends OpenAI.ChatCompletionCreateParamsStreaming {
    provider?: {
        sort: string;
        require_parameters: boolean;
        order?: string[];
    };
}

type LLMMessage = OpenAI.ChatCompletionMessageParam | ToolCallResult;


export class LLMClient {
    private openai: OpenAI;
    private messages: LLMMessage[] = [];
    private mcpClient: MCPClient;
    private apiURL: string;
    private apiKey: string;

    constructor(apiKey: string, apiUrl: string, mcpClient: MCPClient) {
        this.mcpClient = mcpClient;
        this.openai = new OpenAI({
            baseURL: apiUrl,
            apiKey: apiKey,
        });
        this.apiURL = apiUrl;
        this.apiKey = apiKey;
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

    private async makeOpenrouterCall(requestBody: any): Promise<string> {
        try {
            const response = await fetch(`${this.apiURL}/chat/completions`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${this.apiKey}`,
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestBody),
            });

            // Handle successful response first
            if (response.ok) {
                logger.info(await response.text()); // Ensure the response body is read
                const data = (await response.json()) as any;

                // Check for valid content in the successful response
                if (!data.choices || data.choices.length === 0 || !data.choices[0].message?.content) {
                    console.warn('OpenRouter Warning: Response OK but no content generated.', data);
                    throw new Error('OpenRouter response was successful (200 OK) but contained no valid choices or content. This might be due to model warm-up, scaling, or content filtering. Consider retrying, adjusting prompts, or using a different model/provider.');
                }
                // Success: return the content string directly
                return data.choices[0].message.content;
            }

            // --- Error Handling for non-OK responses --- 
            // https://openrouter.ai/docs/api-reference/errors
            let errorCode: number | string = response.status;
            let errorMessage = response.statusText;
            let errorDetails: any = null;

            try {
                logger.info(await response.text()); // Ensure the response body is read
                const errorJson = (await response.json()) as any;
                if (errorJson && errorJson.error) {
                    errorCode = errorJson.error.code || errorCode;
                    errorMessage = errorJson.error.message || errorMessage;
                    errorDetails = { code: errorCode, message: errorMessage, metadata: errorJson.error.metadata };
                    console.error('OpenRouter API Structured Error:', errorDetails);
                } else {
                    errorDetails = errorJson;
                    console.error(`OpenRouter API Error (${response.status}): Non-standard JSON response`, errorDetails);
                }
            } catch (jsonError) {
                try {
                    errorDetails = await response.text();
                } catch (textError) {
                    errorDetails = "<Could not read error body>";
                }
                console.error(`OpenRouter API Error (${response.status}): Raw text response`, errorDetails);
            }

            const finalErrorMessage = `OpenRouter API request failed (${errorCode}): ${errorMessage}`;
            // console.debug(`Openrouter request body that failed: ${JSON.stringify(requestBody, null, 2)}`); 
            throw new Error(finalErrorMessage);

        } catch (error) {
            // Catch fetch errors or errors thrown from response handling
            console.error('Error during _makeOpenrouterCall execution:', error);
            if (error instanceof Error) {
                throw error; // Re-throw the original error
            } else {
                throw new Error('An unknown error occurred during the OpenRouter API call process.');
            }
        }
    }

    async query(params: LLMCallParameters, onData: (chunk: string) => void) {
        let toolCalls: OpenAI.ChatCompletionMessageToolCall[];
        let finishReason: string | null = null;
        let errorCount = 0;
        do {
            toolCalls = [];

            this.dumpMessages();
            let fullQuery: ExtendedChatCompletionParams = {
                model: params.llmModel,
                messages: this.messages,
                stream: true,
                max_tokens: params.maxTokens,
                stop: (params.stopSequence && params.stopSequence.length > 0) ? [params.stopSequence] : undefined,
            };
            (fullQuery as any).provider = {
                sort: "price",
                require_parameters: true,
                order: [
                    "google-vertex/europe",
                    "google-vertex",
                ]
            };
            if (params.schema) {
                fullQuery.response_format = {
                    type: "json_schema",
                    json_schema: {
                        name: "Response",
                        strict: true,
                        schema: zodToJsonSchema(params.schema)
                    }
                };
            }
            else if (params.jsonOutput) {
                fullQuery.response_format = {
                    type: "json_object",
                };
            }
            if (params.reasoning) {
                (fullQuery as any).reasoning = {
                    effort: params.reasoning,
                    exclude: false,
                    maxTokens: 1000,
                }
            }
            try {
                const stream = await this.openai.chat.completions.create(fullQuery as OpenAI.ChatCompletionCreateParamsStreaming);

                let fullResponse: OpenAI.ChatCompletionAssistantMessageParam = {
                    role: "assistant",
                }
                let fullResponseReasoning = "";
                finishReason = null;
                let isReasoning = false;
                for await (const chunk of stream) {
                    // logger.info(JSON.stringify(chunk, null, 2));
                    const delta: any | undefined = chunk.choices[0]?.delta;
                    if (chunk.choices[0]?.finish_reason) {
                        finishReason = chunk.choices[0].finish_reason;
                    }

                    if (delta?.content) {
                        if (!fullResponse.content) {
                            fullResponse.content = "";
                        }
                        fullResponse.content += delta.content;
                        if (isReasoning) {
                            logger.info("</think>");
                            isReasoning = false;
                        }
                        logger.incremental(delta.content);
                        onData(delta.content);
                    }
                    if (delta?.reasoning) {
                        if (!isReasoning) {
                            logger.info("<think>");
                            isReasoning = true;
                        }
                        fullResponseReasoning += delta.reasoning;
                        logger.incremental(delta.reasoning);
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
                if (fullResponseReasoning.length > 0) {
                    this.messages.push({
                        role: "assistant",
                        content: "<think>" + fullResponseReasoning + "</think>",
                    });
                }
                if (fullResponse.content && fullResponse.content.length > 0) {
                    this.messages.push(fullResponse);
                }
                logger.info("Added message:", JSON.stringify(this.messages.at(-1), null, 2));
                logger.info("Finish reason:", finishReason);

                for (const toolCall of toolCalls) {
                    onData(`\n[Calling tool "${toolCall.function.name}"...]\n`);
                    let toolCallResult = await this.mcpClient.callTool(toolCall);
                    logger.info(`Tool call result: ${JSON.stringify(toolCallResult)}`);
                    this.messages.push(toolCallResult);
                }
            } catch (error) {
                logger.error("Error in OpenAI API call:", error);
                errorCount++;
                if (errorCount > 3) {
                    logger.error("Too many errors, stopping.");
                    throw new Error("Too many errors in OpenAI API call.");
                }
                continue;
            }
        } while (toolCalls.length > 0 || finishReason !== "stop" && finishReason !== "length");
        this.dumpMessages();
    }

    private dumpMessages() {
        fs.writeFileSync("saved/messages-dump.text", this.messages.map((message) => `${message.role}:\n${"name" in message ? message.name : ""}\n${message.content}`).join("\n--------------------------------------------------\n"));
    }
}