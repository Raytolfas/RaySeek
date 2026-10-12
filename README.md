# RaySeek

Local AI assistant powered by llama.cpp, Meta FAISS vector memory, live web search, and symbolic math solving.

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
Send a message to the assistant.

Request:
```json
{
  "message": "Hello, who are you?",
  "history": []
}
```

Response:
```json
{
  "response": "..."
}
```

#### GET `/api/health`
Health check.

Response:
```json
{
  "status": "ok"
}
```
