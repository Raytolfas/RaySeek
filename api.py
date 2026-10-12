# Copyright 2026 Raytolfas
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from model_loader import load_engine
from memory_db import query_texts, index_local_files

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm = None
index_local_files()

def get_llm():
    global llm
    if llm is None:
        llm = load_engine()
    return llm

SYSTEM_PROMPT = """Ты — RaySeek 1.0, интеллектуальный ассистент от Raytolfas.
Ты точный, универсальный и лаконичный помощник в программировании, анализе, решении задач и диалоге.
Всегда отвечай строго по существу. Код оформляй в блоки ```язык ... ```."""

class ChatRequest(BaseModel):
    message: str
    history: list = []

@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    cmd = req.message.strip()
    engine = get_llm()

    memory_facts = query_texts(cmd, n_results=1)
    context_snippet = ""
    if memory_facts:
        context_snippet = f"\nКонтекст: {memory_facts[0]}"

    messages = [{"role": "system", "content": SYSTEM_PROMPT + context_snippet}]
    for turn in req.history[-16:]:
        messages.append(turn)
    messages.append({"role": "user", "content": cmd})

    resp = engine.create_chat_completion(
        messages=messages,
        max_tokens=1024,
        temperature=0.6,
        top_p=0.9,
        repeat_penalty=1.15
    )

    reply = resp["choices"][0]["message"]["content"].strip()
    return {"response": reply}

@app.get("/api/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
