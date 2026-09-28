"""Label-wise clustering, graph extraction, cosine retrieval and one-hop expansion."""

import json
import math
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import faiss
import numpy as np
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

import prompts
from data import digest, normalized, read_json, write_json
from model import parse_sentiment, parse_sufficiency


def cluster_rows(rows, vectors, cluster_size=20, max_cluster_size=30, seed=42):
    groups = []
    with threadpool_limits(limits=1):
        for label in (0, 1, 2):
            indices = [i for i, row in enumerate(rows) if row["label"] == label]
            if not indices:
                continue
            count = max(1, math.ceil(len(indices) / cluster_size))
            assignments = KMeans(n_clusters=count, random_state=seed, n_init=10).fit_predict(vectors[indices])
            for cluster_id in sorted(set(assignments)):
                members = [rows[i] for i, c in zip(indices, assignments) if c == cluster_id]
                # Bound extraction output length; never mix labels or drop outliers.
                for start in range(0, len(members), max_cluster_size):
                    groups.append(members[start:start + max_cluster_size])
    return groups


def validate_graph(graph, expected_label):
    if graph.get("sentiment_label") != expected_label:
        raise ValueError("Extracted graph has an inconsistent sentiment label")
    if not isinstance(graph.get("entities"), list) or not isinstance(graph.get("relations"), list):
        raise ValueError("Graph must contain entities and relations lists")
    names = set()
    for entity in graph["entities"]:
        if not all(isinstance(entity.get(k), str) and entity[k].strip() for k in ("name", "type")):
            raise ValueError("Invalid graph entity")
        if not isinstance(entity.get("attributes"), dict):
            raise ValueError("Entity attributes must be an object")
        names.add(normalized(entity["name"]))
    aliases = {"negative": "Bearish", "positive": "Bullish", "bearish": "Bearish",
               "bullish": "Bullish", "neutral": "Neutral"}
    for relation in graph["relations"]:
        if not all(isinstance(relation.get(k), str) and relation[k].strip()
                   for k in ("source", "target", "type", "sentiment")):
            raise ValueError("Invalid graph relation")
        # Appendix B uses a free-form edge sentiment, not a three-value enum.
        # Preserve mixed/descriptive edge sentiments; only final predictions are 0/1/2.
        value = relation["sentiment"].strip()
        relation["sentiment"] = aliases.get(value.lower(), value)
        if type(relation.get("distance")) is not int or relation["distance"] < 1:
            raise ValueError("Every listed relation must have integer distance >= 1, including "
                             "self-loops. Use distance 1 for a direct relationship, even when "
                             "source and target have the same name; never use distance 0")
        for field in ("source", "target"):
            if normalized(relation[field]) not in names:
                # Some valid edges omit an endpoint from the entity list. Preserve
                # the explicit name without inventing its type or other attributes.
                graph["entities"].append({"name": relation[field], "type": "Unspecified", "attributes": {}})
                names.add(normalized(relation[field]))
    return graph


def assemble_graph(parts, metadata):
    entities, relations = [], []
    seen = {}
    for group_id, part in enumerate(parts):
        entities.extend({**entity, "group_id": group_id} for entity in part["entities"])
        for relation in part["relations"]:
            key = tuple(normalized(str(relation[k])) for k in ("source", "target", "type", "sentiment", "distance"))
            if key in seen:
                relations[seen[key]]["group_ids"].append(group_id)
            else:
                seen[key] = len(relations)
                relations.append({**relation, "group_ids": [group_id]})
    if not relations:
        raise ValueError("No relations extracted; cannot build a retrieval index")
    return {"metadata": metadata, "entities": entities, "relations": relations}


def merge_graphs(graphs, datasets):
    """Union source graphs, preserving cluster provenance and shared-node links."""
    if len(graphs) < 2 or graphs.keys() != datasets.keys():
        raise ValueError("Provide at least two graphs and their matching prepared datasets")
    heldout = {normalized(row["text"]) for data in datasets.values() for row in data["evaluation"]}
    sources, entities, relations, seen = {}, [], [], {}
    for name in sorted(graphs):
        graph, data = graphs[name], datasets[name]
        if graph["metadata"].get("data") != data["metadata"]:
            raise ValueError(f"Source graph does not match prepared data: {name}")
        for key, split in (("train_sha256", "train"), ("eval_sha256", "evaluation")):
            if digest(data[split]) != data["metadata"][key]:
                raise ValueError(f"Prepared data checksum mismatch: {name}")
        if any(normalized(row["text"]) in heldout for row in data["train"]):
            raise ValueError(f"Source training text overlaps a held-out split: {name}")
        sources[name] = {"graph_sha256": digest(graph), "data": data["metadata"]}
        entities.extend({**entity, "group_id": f"{name}:{entity['group_id']}"}
                        for entity in graph["entities"])
        for relation in graph["relations"]:
            key = tuple(normalized(str(relation[k]))
                        for k in ("source", "target", "type", "sentiment", "distance"))
            group_ids = [f"{name}:{i}" for i in relation["group_ids"]]
            if key in seen:
                existing = relations[seen[key]]["group_ids"]
                existing.extend(i for i in group_ids if i not in existing)
            else:
                seen[key] = len(relations)
                relations.append({**relation, "group_ids": list(dict.fromkeys(group_ids))})
    if not relations:
        raise ValueError("No relations in source graphs")
    return {"metadata": {"kind": "merged", "sources": sources},
            "entities": entities, "relations": relations}


def build(data, model, output, cluster_size=20, max_cluster_size=30, workers=8):
    output = Path(output)
    metadata = {"data": data["metadata"], "model": model.name, "embedding": model.embedding,
                "seed": model.seed, "cluster_size": cluster_size, "max_cluster_size": max_cluster_size,
                "clustering": "KMeans", "n_init": 10,
                "prompts_sha256": digest([prompts.merge(["TEXT"]), prompts.extract("TEXT", "LABEL")])}
    target = output / "graph.json"
    if target.exists():
        graph = read_json(target)
        if graph["metadata"] != metadata:
            raise ValueError("Graph configuration changed; choose a new --output directory")
        return graph
    print(f"Embedding {len(data['train'])} training texts...", flush=True)
    vectors = model.embed([row["text"] for row in data["train"]])
    groups = cluster_rows(data["train"], vectors, cluster_size, max_cluster_size, model.seed)
    print(f"Extracting {len(groups)} label-homogeneous groups...", flush=True)

    def extract(item):
        i, rows = item
        group_path = output / "groups" / f"{i:05d}.json"
        signature = digest([metadata, rows])
        if group_path.exists():
            saved = read_json(group_path)
            if saved["signature"] != signature:
                raise ValueError("Cluster checkpoint mismatch; use a new output directory")
            return validate_graph(saved["graph"], prompts.LABELS[rows[0]["label"]])
        merged = model.chat(prompts.merge([row["text"] for row in rows]), max_tokens=3500)
        label = prompts.LABELS[rows[0]["label"]]
        prompt = prompts.extract(merged, label)
        for attempt in range(3):
            raw = model.chat(prompt, max_tokens=12000, json_output=True)
            try:
                graph = validate_graph(json.loads(raw), label)
                break
            except (ValueError, KeyError, TypeError) as error:
                if attempt == 2:
                    raise
                prompt = (prompts.extract(merged, label) +
                          f"\nThe previous JSON failed validation: {error}.\n"
                          "Correct the invalid structure while preserving the text's facts. "
                          "Output only the complete corrected JSON. Previous JSON:\n" + raw)
        write_json(group_path, {"signature": signature, "source_ids": [r["id"] for r in rows],
                               "label": rows[0]["label"], "merged_text": merged, "graph": graph,
                               "schema_repair_attempts": attempt})
        return graph

    parts = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for part in pool.map(extract, enumerate(groups)):
            parts.append(part)
            if len(parts) % 10 == 0 or len(parts) == len(groups):
                print(f"Graphs: {len(parts)}/{len(groups)}", flush=True)
    graph = assemble_graph(parts, metadata)
    write_json(target, graph)
    return graph


def sentence(relation):
    return f"{relation['source']} {relation['type']} {relation['target']}. Sentiment: {relation['sentiment']}."


class Retriever:
    def __init__(self, graph, vectors):
        self.relations = graph["relations"]
        self.sentences = [sentence(r) for r in self.relations]
        self.vectors = np.ascontiguousarray(vectors, dtype="float32")
        if len(self.vectors) != len(self.relations):
            raise ValueError("Embedding/graph size mismatch")
        faiss.omp_set_num_threads(1)
        self.index = faiss.IndexFlatIP(self.vectors.shape[1])
        self.index.add(self.vectors)
        self.adjacency = defaultdict(set)
        for i, relation in enumerate(self.relations):
            if relation["distance"] == 1:
                for field in ("source", "target"):
                    self.adjacency[normalized(relation[field])].add(i)

    def retrieve(self, vector, top_k=5):
        _, indices = self.index.search(np.asarray([vector], dtype="float32"), min(top_k, len(self.relations)))
        return indices[0].tolist()

    def expand(self, indices, vector, max_neighbors=20):
        candidates = set()
        for i in indices:
            for field in ("source", "target"):
                candidates.update(self.adjacency[normalized(self.relations[i][field])])
        candidates.difference_update(indices)
        # Freeze the frontier: this is exactly one hop, not recursive expansion.
        ranked = sorted(candidates, key=lambda i: (-float(self.vectors[i] @ vector), i))
        return indices + ranked[:max_neighbors]

    def context(self, indices):
        return "\n".join(self.sentences[i] for i in indices)


def predict(query, vector, retriever, model, top_k=5, max_neighbors=20):
    indices = retriever.retrieve(vector, top_k)
    judgments = []
    for stage in ("retrieval", "one_hop"):
        context = retriever.context(indices)
        judgment = model.chat(prompts.sufficient(query, context), max_tokens=300)
        enough = parse_sufficiency(judgment)
        judgments.append({"stage": stage, "sufficient": enough, "response": judgment,
                          "edge_ids": list(indices)})
        if enough:
            response = model.chat(prompts.answer(query, context), max_tokens=64)
            return {"prediction": parse_sentiment(response), "route": stage,
                    "judgments": judgments, "response": response}
        if stage == "retrieval":
            expanded = retriever.expand(indices, vector, max_neighbors)
            if expanded == indices:
                break
            indices = expanded
    response = model.chat(prompts.answer(query), max_tokens=64)
    return {"prediction": parse_sentiment(response), "route": "direct",
            "judgments": judgments, "response": response}
