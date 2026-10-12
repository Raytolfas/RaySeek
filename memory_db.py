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

def query_texts(query, n_results=2):
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

def index_local_files():
    if not os.path.exists("brand_data"):
        os.makedirs("brand_data")
        return
    _init_storage()
    known_texts = {d["text"] for d in _documents}
    new_texts = []
    new_metas = []
    for filename in os.listdir("brand_data"):
        if filename.endswith(".txt"):
            filepath = os.path.join("brand_data", filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                paragraphs = [p.strip() for p in content.split("\n") if len(p.strip()) > 10]
                for p in paragraphs:
                    if p not in known_texts:
                        new_texts.append(p)
                        new_metas.append({"source": filename})
                        known_texts.add(p)
            except Exception:
                pass
    if new_texts:
        add_texts(new_texts, new_metas)
