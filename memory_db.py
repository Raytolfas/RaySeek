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
import json
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel, logging as tf_logging
import torch

tf_logging.set_verbosity_error()

FAISS_INDEX_FILE = "rayseek_faiss.index"
METADATA_FILE = "rayseek_faiss_meta.json"
EMBED_MODEL_NAME = "cointegrated/rubert-tiny2"

_tokenizer = None
_model = None
_index = None
_documents = []

def _get_embed_model():
    global _tokenizer, _model
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
        _model = AutoModel.from_pretrained(EMBED_MODEL_NAME)
        _model.eval()
    return _tokenizer, _model

def _embed_texts(texts):
    tok, mod = _get_embed_model()
    inputs = tok(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        out = mod(**inputs)
        embeddings = out.last_hidden_state[:, 0, :]
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings.numpy().astype("float32")

def _init_storage():
    global _index, _documents
    if _index is None:
        dim = 312
        if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(METADATA_FILE):
            _index = faiss.read_index(FAISS_INDEX_FILE)
            with open(METADATA_FILE, "r", encoding="utf-8") as f:
                _documents = json.load(f)
        else:
            _index = faiss.IndexFlatIP(dim)
            _documents = []

def save_index():
    global _index, _documents
    if _index is not None:
        faiss.write_index(_index, FAISS_INDEX_FILE)
        with open(METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(_documents, f, ensure_ascii=False)

def add_texts(texts, metadatas=None):
    global _index, _documents
    _init_storage()
    if not texts:
        return
    vectors = _embed_texts(texts)
    _index.add(vectors)
    for i, t in enumerate(texts):
        meta = metadatas[i] if metadatas and i < len(metadatas) else {}
        _documents.append({"text": t, "meta": meta})
    save_index()

def query_texts(query, n_results=3):
    global _index, _documents
    _init_storage()
    if _index.ntotal == 0:
        return []
    vec = _embed_texts([query])
    k = min(n_results, _index.ntotal)
    distances, indices = _index.search(vec, k)
    results = []
    for idx in indices[0]:
        if idx != -1 and idx < len(_documents):
            results.append(_documents[idx]["text"])
    return results

def get_doc_count():
    _init_storage()
    return _index.ntotal if _index is not None else 0

def extract_file_content(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".pdf":
        try:
            from pypdf import PdfReader
            reader = PdfReader(filepath)
            return "\n".join([page.extract_text() or "" for page in reader.pages])
        except Exception:
            return ""
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""

def chunk_text(text, chunk_size=350):
    lines = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    curr = ""
    for line in lines:
        if len(curr) + len(line) < chunk_size:
            curr += ("\n" if curr else "") + line
        else:
            if curr:
                chunks.append(curr)
            curr = line
    if curr:
        chunks.append(curr)
    return chunks

def index_path(target_path):
    _init_storage()
    if not os.path.exists(target_path):
        return 0

    known_texts = {d["text"] for d in _documents}
    supported = {".txt", ".md", ".py", ".js", ".ts", ".html", ".css", ".json", ".csv", ".pdf", ".cpp", ".c", ".h", ".rs", ".go"}
    files_to_scan = []

    if os.path.isfile(target_path):
        files_to_scan.append(target_path)
    else:
        for root, _, files in os.walk(target_path):
            if any(ign in root for ign in [".git", ".venv", "node_modules", "__pycache__", "dist", "build"]):
                continue
            for f in files:
                if os.path.splitext(f)[1].lower() in supported:
                    files_to_scan.append(os.path.join(root, f))

    new_texts = []
    new_metas = []

    for fpath in files_to_scan:
        content = extract_file_content(fpath)
        if not content:
            continue
        rel = os.path.relpath(fpath, os.getcwd())
        chunks = chunk_text(content)
        for chunk in chunks:
            if len(chunk) > 20 and chunk not in known_texts:
                doc_text = f"[{rel}]:\n{chunk}"
                new_texts.append(doc_text)
                new_metas.append({"source": rel})
                known_texts.add(chunk)

    if new_texts:
        add_texts(new_texts, new_metas)
    return len(new_texts)

def index_local_files():
    if not os.path.exists("brand_data"):
        os.makedirs("brand_data")
        return
    index_path("brand_data")
