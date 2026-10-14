# RaySeek 1.1

Local AI assistant powered by llama.cpp, Meta FAISS vector memory, deep Chain of Thought reasoning, autonomous tool execution, live web search, and symbolic math solving.

## Features in Version 1.1
- **Deep Reasoning Stream**: Dynamic `<think>` chain-of-thought analysis for complex queries and code architecture.
- **Universal Local RAG**: `/load <path>` indexes entire project folders, source code, text documents, and PDFs into Meta FAISS C++ vector memory.
- **Autonomous Tool Use**: Native system shell (`/bash`), file system reading/writing, web search, and time inspection.
- **Adaptive 8K Context**: Dynamic memory scaling ensuring fast inference on standard consumer GPUs.

## Installation

```bash
git clone https://github.com/Raytolfas/RaySeek.git
cd RaySeek
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> Model weights (`Qwen2.5-3B-Instruct-GGUF`) download automatically from Hugging Face Hub on first run.

## CLI Usage

```bash
python main.py
```

### Commands
- `/load <path>` — index local directory or file (code, txt, pdf) into vector memory
- `/think` — toggle deep reasoning visibility
- `/bash <cmd>` — run shell command
- `/web <query>` — search the web in real-time
- `/translate <text>` — translate text RU <-> EN
- `/info` — system status and context statistics
- `/help` — display commands list
- `/exit` — terminate session

## API Usage

```bash
python api.py
```

The server starts on `http://127.0.0.1:8000`.

### Endpoints

#### POST `/api/chat`
Send a message to the assistant with autonomous tool execution.

Request:
```json
{
  "message": "Hello, who are you?",
  "history": []
}
```

#### POST `/api/load`
Index a file or folder into the local FAISS memory.

Request:
```json
{
  "path": "./my_project"
}
```

#### GET `/api/info`
Retrieve system and model specifications.

#### GET `/api/health`
Health check.
