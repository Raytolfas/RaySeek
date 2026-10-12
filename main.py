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

import sys
import re
import random
import sympy
from model_loader import load_engine
from memory_db import query_texts, get_doc_count, index_local_files
from search import web_search
from deep_translator import GoogleTranslator

SYSTEM_PROMPT = """Ты — RaySeek 1.0, интеллектуальный ассистент от Raytolfas.
Ты точный, универсальный и лаконичный помощник в программировании, анализе, решении задач и диалоге.
Всегда отвечай строго по существу. Код оформляй в блоки ```язык ... ```."""

def format_math_response(text):
    clean_text = text.replace("^", "**")
    match = re.search(r"([\d\+\-\*\/\(\)\s\.\,\^]|sqrt|sin|cos|tan|log|pi|e)+", clean_text)
    if not match:
        return None

    candidate = match.group(0).strip()
    candidate = re.sub(r"^[\+\-\*\/\s\.\,\^]+", "", candidate)
    candidate = re.sub(r"[\+\-\*\/\s\.\,\^\=\?]+$", "", candidate)

    if not re.search(r"[\+\-\*\/\(\)\^]|sqrt|sin|cos|tan|log", candidate):
        return None

    try:
        expr = sympy.sympify(candidate)
        res = expr.evalf()
        if res == int(res):
            res = int(res)
        else:
            res = round(float(res), 4)

        templates = [
            f"{candidate} = {res}",
            f"Ответ: {res}",
            f"{res}"
        ]
        return random.choice(templates)
    except Exception:
        return None

def translate_text(text):
    has_cyrillic = bool(re.search(r"[а-яА-ЯёЁ]", text))
    target_lang = "en" if has_cyrillic else "ru"
    try:
        return GoogleTranslator(source="auto", target=target_lang).translate(text)
    except Exception as e:
        return f"Error: {e}"

def print_help():
    print("Commands:")
    print("  /web <query>       Search the web in real-time")
    print("  /translate <text>  Translate RU <-> EN")
    print("  /info              System info")
    print("  /help              Show this help")
    print("  /exit              Exit session")

def main():
    try:
        index_local_files()
        llm = load_engine()
    except Exception as e:
        print(f"Init error: {e}")
        sys.exit(1)

    ctx_size = getattr(llm, "n_ctx", lambda: 8192)() if callable(getattr(llm, "n_ctx", None)) else 8192
    print(f"RaySeek 1.0 (Context: {ctx_size} tokens)")
    print("Type /help for commands, /exit to quit.\n")

    history = []

    while True:
        try:
            line = input("> ").strip()
            if not line:
                continue

            low = line.lower()

            if low in ("/exit", "/quit", "exit", "quit"):
                break

            if low == "/help":
                print_help()
                continue

            if low == "/info":
                docs = get_doc_count()
                print("Model: Qwen2.5-3B-Instruct (GGUF Q4_K_M)")
                print("Engine: llama.cpp")
                print(f"Context window: {ctx_size} tokens")
                print(f"Memory: Meta FAISS ({docs} items)")
                continue

            if low.startswith("/translate "):
                text = line[11:].strip()
                if not text:
                    continue
                res = translate_text(text)
                print(res)
                continue

            if low.startswith("/web "):
                query = line[5:].strip()
                res = web_search(query, max_results=3)
                web_context = "\n".join([f"- {r['title']}: {r['body']}" for r in res]) if res else "No results."
                line = f"Question: {query}\nWeb search:\n{web_context}"

            math_response = format_math_response(line)
            if math_response:
                print(math_response)
                history.append({"role": "user", "content": line})
                history.append({"role": "assistant", "content": math_response})
                continue

            memory_facts = query_texts(line, n_results=2)
            context_snippet = ""
            if memory_facts:
                context_snippet = "\n\nFacts:\n" + "\n".join(memory_facts)

            messages = [{"role": "system", "content": SYSTEM_PROMPT + context_snippet}]
            for turn in history[-16:]:
                messages.append(turn)
            messages.append({"role": "user", "content": line})

            stream = llm.create_chat_completion(
                messages=messages,
                max_tokens=1024,
                temperature=0.65,
                top_p=0.9,
                repeat_penalty=1.18,
                stream=True
            )

            full_reply = ""
            for chunk in stream:
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    print(content, end="", flush=True)
                    full_reply += content

            print()

            history.append({"role": "user", "content": line})
            history.append({"role": "assistant", "content": full_reply})

        except KeyboardInterrupt:
            print("\nExiting.")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
