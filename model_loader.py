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
import psutil
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

REPO_ID = "Qwen/Qwen2.5-3B-Instruct-GGUF"
FILENAME = "qwen2.5-3b-instruct-q4_k_m.gguf"
MODEL_DIR = "models"
GGUF_PATH = os.path.join(MODEL_DIR, FILENAME)

def get_optimal_context(target_ctx=8192):
    try:
        mem = psutil.virtual_memory()
        available_mb = mem.available / (1024 * 1024)
        if available_mb < 2000:
            return 2048
        elif available_mb < 4000:
            return 4096
        return min(target_ctx, 16384)
    except Exception:
        return 4096

def load_engine(n_ctx=None):
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(GGUF_PATH):
        hf_hub_download(
            repo_id=REPO_ID,
            filename=FILENAME,
            local_dir=MODEL_DIR,
            local_dir_use_symlinks=False
        )

    if n_ctx is None:
        n_ctx = get_optimal_context(8192)

    threads = max(1, os.cpu_count() - 1)
    llm = Llama(
        model_path=GGUF_PATH,
        n_ctx=n_ctx,
        n_threads=threads,
        n_gpu_layers=-1,
        verbose=False
    )
    return llm
