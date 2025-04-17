# LLM Text Adventure

LLM Text Adventure is an interactive text adventure game with LLM-powered storytelling and image generation capabilities.

The project is on early stages of development and is meant to be used as a reference.

## Features

- **Multi-agent architecture**: Different tasks performed by different AI agents, making it possible to use best matching model for every task.
- **Memory System**: The adventure remembers your choices and builds a coherent narrative

## Prerequisites

- Node.js v18+ and npm
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) for image generation
- An [OpenRouter](https://openrouter.ai/) API key for accessing language models

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/ArcticFox8515/LLMTextAdventure.git
cd LLMTextAdventure
```

### 2. Install dependencies

```bash
npm install
```

### 3. Set up environment variables

Copy the example environment file and update it with your settings:

```bash
cp .env.example .env
```

Edit the `.env` file and update the following variables:

```
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_MODEL=your_preferred_model
OPENROUTER_MODEL_MEMORY_FETCH=your_preferred_memory_model
OPENROUTER_MODEL_NARRATIVE=your_preferred_narrative_model
```

Available models and their capabilities can be found on [OpenRouter's website](https://openrouter.ai/models).

### 4. Set up story parameters

Create `prompts/story/story-parameters.yaml` and fill up with story starting parameters. You can use one of example files:
* `prompts/story/story-parameters.eldoria.yaml`
* `prompts/story/story-parameters.cyberpunk.yaml`
or write your own story

## Running the Application

Run both the client and server in development mode:

```bash
npm run build
npm run dev
```

This will:
1. Build the server
2. Start the client development server
3. Start the backend server

You can then open http://127.0.0.1:3001/ in your browser

## Image Generation

MCP Adventure uses ComfyUI for image generation. Make sure ComfyUI is running and accessible at the URL specified in your `.env` file (default: `http://127.0.0.1:8190`). Make sure the checkpoint from `story-parameters.xml` is present in the model.

You can edit image generator prompt located in `prompts/image-generator-prompt.json`