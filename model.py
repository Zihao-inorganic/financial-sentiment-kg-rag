"""OpenAI calls with persistent caching and response parsing."""

import json
import re
import sqlite3
import threading
import zlib
from pathlib import Path

import numpy as np
from openai import OpenAI

from data import digest


def parse_sentiment(text):
    matches = re.findall(r"Sentiment\s*:\s*\[?([012])\]?(?![\d/])", text, re.I)
    if not matches:
        raise ValueError(f"Missing sentiment label in response: {text[:160]!r}")
    return int(matches[-1])


def parse_sufficiency(text):
    match = re.match(r"\s*(?:\*\*|['\"])?(yes|no)\b", text, re.I)
    if not match:
        raise ValueError(f"Missing Yes/No judgment: {text[:160]!r}")
    return match.group(1).lower() == "yes"


class Model:
    def __init__(self, model="gpt-4o-mini-2024-07-18", embedding="text-embedding-ada-002",
                 cache=".cache/api.sqlite", seed=42):
        self.name, self.embedding, self.seed = model, embedding, seed
        self.client = OpenAI(max_retries=5, timeout=120)
        self.endpoint = str(self.client.base_url)
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(cache, check_same_thread=False, timeout=60)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value BLOB)")
        self.db.commit()
        self.lock = threading.Lock()

    def get(self, key):
        with self.lock:
            row = self.db.execute("SELECT value FROM cache WHERE key=?", (key,)).fetchone()
        return json.loads(zlib.decompress(row[0])) if row else None

    def put(self, key, value):
        blob = zlib.compress(json.dumps(value).encode())
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO cache VALUES (?, ?)", (key, blob))
            self.db.commit()

    def chat(self, prompt, max_tokens=1200, json_output=False):
        request = {"model": self.name, "messages": [{"role": "user", "content": prompt}],
                   "temperature": 0, "seed": self.seed, "max_tokens": max_tokens}
        if json_output:
            request["response_format"] = {"type": "json_object"}
        key = digest([self.endpoint, "chat", request])
        record = self.get(key)
        if record is None:
            for attempt in range(3):
                response = self.client.chat.completions.create(**request)
                choice = response.choices[0]
                if choice.finish_reason == "stop" and choice.message.content:
                    break
                if choice.finish_reason != "length" or request["max_tokens"] >= 16384:
                    raise RuntimeError(f"Incomplete model response ({choice.finish_reason})")
                request = {**request, "max_tokens": min(16384, max(512, request["max_tokens"] * 2))}
            else:
                raise RuntimeError("Response still truncated after two output-budget increases")
            record = {"text": choice.message.content, "model": response.model,
                      "system_fingerprint": response.system_fingerprint,
                      "usage": response.usage.model_dump(), "request": request,
                      "length_retries": attempt}
            self.put(key, record)
        return record["text"]

    def embed(self, texts, batch_size=64):
        if not texts:
            raise ValueError("Cannot embed an empty list")
        keys = [digest([self.endpoint, "embedding", self.embedding, text]) for text in texts]
        records = [self.get(key) for key in keys]
        missing = [i for i, record in enumerate(records) if record is None]
        for start in range(0, len(missing), batch_size):
            indices = missing[start:start + batch_size]
            response = self.client.embeddings.create(model=self.embedding,
                input=[texts[i] for i in indices], encoding_format="float")
            if sorted(item.index for item in response.data) != list(range(len(indices))):
                raise RuntimeError("Embedding response has missing or duplicate indices")
            for item in response.data:
                i = indices[item.index]
                records[i] = {"vector": item.embedding, "model": response.model}
                self.put(keys[i], records[i])
        vectors = np.asarray([record["vector"] for record in records], dtype="float32")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if not np.isfinite(vectors).all() or (norms <= 0).any():
            raise ValueError("Invalid embedding vector")
        return np.ascontiguousarray(vectors / norms)
