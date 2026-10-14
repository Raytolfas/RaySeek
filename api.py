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

import re
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from model_loader import load_engine
from memory_db import query_texts, index_local_files, index_path, get_doc_count
import tools

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

SYSTEM_PROMPT = """Ты — RaySeek 1.1, интеллектуальный ассистент от Raytolfas.
Ты точный, универсальный и мыслящий помощник в программировании, анализе и решении задач.
Перед сложными рассуждениями ты используешь блок <think>...</think>, затем даешь идеальный ответ.
При необходимости ты можешь запросить инструмент: <tool>bash: команда</tool>, <tool>read_file: путь</tool>, <tool>write_file: путь|контент</tool>, <tool>list_dir: папка</tool>, <tool>time</tool>, <tool>web: запрос</tool>.
Весь код оформляй в блоки ```язык ... ```."""

class ChatRequest(BaseModel):
    message: str
    history: list = []

class LoadRequest(BaseModel):
    path: str

@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    cmd = req.message.strip()
    engine = get_llm()

    memory_facts = query_texts(cmd, n_results=3)
    context_snippet = ""
    if memory_facts:
        context_snippet = "\n\nКонтекст из памяти:\n" + "\n".join(memory_facts)

    messages = [{"role": "system", "content": SYSTEM_PROMPT + context_snippet}]
    for turn in req.history[-16:]:
        messages.append(turn)
    messages.append({"role": "user", "content": cmd})

    resp = engine.create_chat_completion(
        messages=messages,
        max_tokens=1500,
        temperature=0.6,
        top_p=0.9,
        repeat_penalty=1.15
    )

    reply = resp["choices"][0]["message"]["content"].strip()

    tool_match = re.search(r"<tool>([a-zA-Z_]+)(?::\s*([^<]+))?</tool>", reply)
    tool_output = None
    if tool_match:
        tool_name = tool_match.group(1)
        tool_arg = tool_match.group(2) if tool_match.group(2) else ""
        tool_output = tools.execute_tool_call(tool_name, tool_arg)

        tool_messages = list(messages)
        tool_messages.append({"role": "assistant", "content": reply})
        tool_messages.append({
            "role": "user",
            "content": f"[Результат инструмента {tool_name}]:\n{tool_output}\n\nСформулируй итоговый ответ."
        })

        followup = engine.create_chat_completion(
            messages=tool_messages,
            max_tokens=1500,
            temperature=0.6,
            top_p=0.9,
            repeat_penalty=1.15
        )
        final_text = followup["choices"][0]["message"]["content"].strip()
        return {
            "response": final_text,
            "raw_reply": reply,
            "tool_used": tool_name,
            "tool_output": tool_output
        }

    return {"response": reply}

@app.post("/api/load")
def load_endpoint(req: LoadRequest):
    count = index_path(req.path)
    return {
        "status": "ok",
        "new_chunks": count,
        "total_documents": get_doc_count(),
        "path": req.path
    }

@app.get("/api/info")
def info_endpoint():
    engine = get_llm()
    ctx_size = getattr(engine, "n_ctx", lambda: 8192)() if callable(getattr(engine, "n_ctx", None)) else 8192
    return {
        "model": "Qwen2.5-3B-Instruct (GGUF Q4_K_M)",
        "version": "1.1",
        "context_window": ctx_size,
        "indexed_chunks": get_doc_count()
    }

@app.get("/api/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
