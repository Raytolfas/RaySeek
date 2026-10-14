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
from memory_db import query_texts, get_doc_count, index_local_files, index_path
from search import web_search
from deep_translator import GoogleTranslator
import tools

SYSTEM_PROMPT = """Ты — RaySeek 1.1, интеллектуальный ассистент нового поколения от экосистемы Raytolfas.
Ты точный, глубоко мыслящий инженерный помощник в программировании, анализе и решении задач.

Твои возможности и правила:
1. Рассуждение (Thinking): Перед решением сложных задач, кода или рассуждений ты формируешь блок <think>
...твои рассуждения, план действий и анализ...
</think>
После чего выдаешь чистый, идеальный итоговый ответ.

2. Автономные инструменты (Tools): При необходимости ты можешь запросить выполнение системного инструмента:
- Выполнить команду: <tool>bash: команда</tool>
- Прочитать файл: <tool>read_file: путь_к_файлу</tool>
- Записать файл: <tool>write_file: путь|содержимое</tool>
- Список файлов: <tool>list_dir: папка</tool>
- Системное время: <tool>time</tool>
- Поиск в сети: <tool>web: запрос</tool>

3. Код: Весь программный код всегда оформляй в блоки ```язык ... ```."""

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
    print("  /load <path>       Index local files or folder (code, txt, pdf) into memory")
    print("  /think             Toggle deep reasoning display")
    print("  /bash <cmd>        Execute system shell command")
    print("  /web <query>       Search the web in real-time")
    print("  /translate <text>  Translate RU <-> EN")
    print("  /info              System status and context info")
    print("  /help              Show this help")
    print("  /exit              Exit session")

def execute_model_stream(llm, messages, show_think=True):
    stream = llm.create_chat_completion(
        messages=messages,
        max_tokens=1500,
        temperature=0.6,
        top_p=0.9,
        repeat_penalty=1.15,
        stream=True
    )

    full_reply = ""
    in_think = False

    for chunk in stream:
        delta = chunk["choices"][0].get("delta", {})
        content = delta.get("content", "")
        if not content:
            continue

        full_reply += content

        if "<think>" in full_reply and not in_think:
            in_think = True
            if show_think:
                print("\n[Thinking...]\n", end="", flush=True)

        if in_think:
            if "</think>" in full_reply:
                in_think = False
                if show_think:
                    print("\n[Thinking complete]\n", end="", flush=True)
            elif show_think:
                print(content, end="", flush=True)
        else:
            print(content, end="", flush=True)

    print()
    return full_reply

def main():
    try:
        index_local_files()
        llm = load_engine()
    except Exception as e:
        print(f"Init error: {e}")
        sys.exit(1)

    ctx_size = getattr(llm, "n_ctx", lambda: 8192)() if callable(getattr(llm, "n_ctx", None)) else 8192
    print(f"RaySeek 1.1 (Context: {ctx_size} tokens | RAG & Reasoning Active)")
    print("Type /help for commands, /exit to quit.\n")

    history = []
    show_think = True

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

            if low == "/think":
                show_think = not show_think
                status = "shown" if show_think else "hidden"
                print(f"[Deep reasoning stream: {status}]")
                continue

            if low == "/info":
                docs = get_doc_count()
                t_status = "active" if show_think else "hidden"
                print("Model: Qwen2.5-3B-Instruct (GGUF Q4_K_M)")
                print("Engine: llama.cpp AVX2")
                print(f"Context window: {ctx_size} tokens")
                print(f"Memory: Meta FAISS ({docs} indexed chunks)")
                print(f"Reasoning stream: {t_status}")
                print("Tools: bash, web, read/write_file, list_dir, time")
                continue

            if low.startswith("/load "):
                target_path = line[6:].strip()
                count = index_path(target_path)
                print(f"[Loaded {count} new chunks from '{target_path}' into FAISS memory. Total: {get_doc_count()}]")
                continue

            if low.startswith("/bash "):
                cmd_to_run = line[6:].strip()
                res = tools.run_bash(cmd_to_run)
                print(res)
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

            memory_facts = query_texts(line, n_results=3)
            context_snippet = ""
            if memory_facts:
                context_snippet = "\n\nIndexed knowledge:\n" + "\n".join(memory_facts)

            messages = [{"role": "system", "content": SYSTEM_PROMPT + context_snippet}]
            for turn in history[-16:]:
                messages.append(turn)
            messages.append({"role": "user", "content": line})

            full_reply = execute_model_stream(llm, messages, show_think=show_think)

            tool_match = re.search(r"<tool>([a-zA-Z_]+)(?::\s*([^<]+))?</tool>", full_reply)
            if tool_match:
                tool_name = tool_match.group(1)
                tool_arg = tool_match.group(2) if tool_match.group(2) else ""
                print(f"\n[Executing Tool: {tool_name} ...] ", end="", flush=True)
                tool_result = tools.execute_tool_call(tool_name, tool_arg)
                print("Done.\n")

                tool_messages = list(messages)
                tool_messages.append({"role": "assistant", "content": full_reply})
                tool_messages.append({
                    "role": "user",
                    "content": f"[Tool Result for {tool_name}]:\n{tool_result}\n\nСформулируй итоговый ответ с учетом этих данных."
                })

                followup_reply = execute_model_stream(llm, tool_messages, show_think=show_think)
                full_reply = f"{full_reply}\n\n[Tool Output]:\n{tool_result}\n\n{followup_reply}"

            history.append({"role": "user", "content": line})
            history.append({"role": "assistant", "content": full_reply})

        except KeyboardInterrupt:
            print("\nExiting.")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
