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

import os
import sys
import subprocess
import datetime
from search import web_search
import memory_db

def run_bash(cmd, timeout=30):
    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        out = proc.stdout.strip()
        err = proc.stderr.strip()
        if proc.returncode == 0:
            return out if out else "[Command completed successfully with no output]"
        return f"Exit code {proc.returncode}:\n{err if err else out}"
    except subprocess.TimeoutExpired:
        return f"Timeout: command took longer than {timeout} seconds"
    except Exception as e:
        return f"Execution error: {e}"

def read_file(path, max_lines=200):
    try:
        if not os.path.exists(path):
            return f"File not found: {path}"
        if os.path.isdir(path):
            return f"{path} is a directory, not a file"
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = [f.readline() for _ in range(max_lines)]
        return "".join(lines)
    except Exception as e:
        return f"Read error: {e}"

def write_file(path, content):
    try:
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"File successfully written: {path} ({len(content)} chars)"
    except Exception as e:
        return f"Write error: {e}"

def list_dir(path="."):
    try:
        if not os.path.exists(path):
            return f"Path does not exist: {path}"
        items = os.listdir(path)
        return "\n".join(items[:100]) if items else "[Empty directory]"
    except Exception as e:
        return f"List error: {e}"

def get_system_time():
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S (%A)")

def execute_tool_call(tool_name, argument):
    name = tool_name.strip().lower()
    arg = argument.strip()

    if name in ("bash", "run_bash", "cmd", "shell"):
        return run_bash(arg)
    elif name in ("web", "web_search", "search"):
        results = web_search(arg, max_results=3)
        if not results:
            return "No web results found."
        return "\n".join([f"- {r['title']}: {r['body']}" for r in results])
    elif name in ("read_file", "cat"):
        return read_file(arg)
    elif name in ("write_file", "save_file"):
        parts = arg.split("|", 1)
        if len(parts) == 2:
            return write_file(parts[0].strip(), parts[1].strip())
        return "Write file syntax requires: filename|content"
    elif name in ("list_dir", "ls", "dir"):
        return list_dir(arg if arg else ".")
    elif name in ("time", "now"):
        return get_system_time()
    elif name in ("load", "index_path"):
        count = memory_db.index_path(arg)
        return f"Indexed {count} chunks from {arg} into memory."
    return f"Unknown tool: {tool_name}"
