"""CoT and KG-RAG evaluation with retrieval/fallback statistics."""

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

import prompts
from data import digest, read_json, write_json
from graph import Retriever, predict, sentence
from model import parse_sentiment


def metrics(rows, method):
    if not rows:
        return {"n": 0, "accuracy": None, "macro_f1": None, "confusion_matrix": None}
    gold = [r["label"] for r in rows]
    predicted = [r[method]["prediction"] for r in rows]
    return {"n": len(rows), "accuracy": accuracy_score(gold, predicted),
            "macro_f1": f1_score(gold, predicted, labels=[0, 1, 2], average="macro", zero_division=0),
            "confusion_matrix": confusion_matrix(gold, predicted, labels=[0, 1, 2]).tolist()}


def summarize(rows, metadata):
    result = {"metadata": metadata, "label_counts": dict(Counter(r["label"] for r in rows)),
              "metrics": {method: metrics(rows, method) for method in ("cot", "kg")}}
    routes = Counter(row["kg"]["route"] for row in rows)
    result["routes"] = {route: {"fraction": routes[route] / len(rows),
                              **metrics([r for r in rows if r["kg"]["route"] == route], "kg")}
                        for route in ("retrieval", "one_hop", "direct")}
    rag_rows = [r for r in rows if r["kg"]["route"] != "direct"]
    result["rag_combined"] = {"fraction": len(rag_rows) / len(rows), **metrics(rag_rows, "kg")}
    return result


def evaluate(data, graph, model, output, limit=None, workers=8, top_k=5, max_neighbors=20):
    output = Path(output)
    rows = data["evaluation"]
    graph_data = graph["metadata"].get("data")
    if graph["metadata"].get("kind") == "merged":
        source = graph["metadata"]["sources"].get(data["metadata"]["dataset"])
        if source is None or source["data"] != data["metadata"]:
            raise ValueError("Evaluation dataset is not a matching merged-graph source")
        graph_data = source["data"]
    if graph_data is None or graph_data["train_sha256"] != data["metadata"]["train_sha256"]:
        raise ValueError("Graph and evaluation split do not match")
    if limit and limit < len(rows):
        rows, _ = train_test_split(rows, train_size=limit, random_state=model.seed,
                                   stratify=[r["label"] for r in rows])
    metadata = {"dataset": data["metadata"]["dataset"], "model": model.name,
                "embedding": model.embedding, "seed": model.seed, "top_k": top_k,
                "max_neighbors": max_neighbors, "eval_count": len(rows),
                "full_eval_count": len(data["evaluation"]), "graph_sha256": digest(graph),
                "eval_sha256": digest(rows), "temperature": 0,
                "prompts_sha256": digest([prompts.cot("QUERY"), prompts.answer("QUERY"),
                                           prompts.answer("QUERY", "CONTEXT"), prompts.sufficient("QUERY", "CONTEXT")])}
    expected = {row["id"]: row for row in rows}
    if len(expected) != len(rows):
        raise ValueError("Duplicate IDs in evaluation data")
    manifest = output / "evaluation.json"
    prediction_path = output / "predictions.jsonl"
    if prediction_path.exists() and not manifest.exists():
        raise ValueError("Evaluation checkpoint has no configuration manifest")
    if manifest.exists() and read_json(manifest) != metadata:
        raise ValueError("Evaluation configuration changed; choose a new --output directory")
    saved = {}
    if prediction_path.exists():
        for line in prediction_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["id"] in saved or row["id"] not in expected:
                raise ValueError("Duplicate or unexpected row in evaluation checkpoint")
            if any(row[k] != expected[row["id"]][k] for k in ("text", "label")):
                raise ValueError("Checkpoint text or label does not match evaluation data")
            for method in ("cot", "kg"):
                prediction = row[method]["prediction"]
                if type(prediction) is not int or prediction not in (0, 1, 2):
                    raise ValueError("Invalid checkpoint prediction")
                if prediction != parse_sentiment(row[method]["response"]):
                    raise ValueError("Checkpoint prediction does not match model response")
            if row["kg"]["route"] not in ("retrieval", "one_hop", "direct"):
                raise ValueError("Invalid checkpoint route")
            saved[row["id"]] = row
    write_json(manifest, metadata)
    print(f"Embedding {len(graph['relations'])} graph edges and {len(rows)} queries...", flush=True)
    retriever = Retriever(graph, model.embed([sentence(r) for r in graph["relations"]]))
    vectors = model.embed([row["text"] for row in rows])

    def score(item):
        i, row = item
        # Only the query text reaches prediction; evaluation labels never enter prompts.
        query = row["text"]
        cot = model.chat(prompts.cot(query), max_tokens=1200)
        kg = predict(query, vectors[i], retriever, model, top_k, max_neighbors)
        return {**row, "cot": {"prediction": parse_sentiment(cot), "response": cot},
                "kg": kg}

    with prediction_path.open("a", encoding="utf-8") as stream, ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(score, (i, row)): row["id"] for i, row in enumerate(rows) if row["id"] not in saved}
        for future in as_completed(pending):
            try:
                result = future.result()
            except Exception as error:
                raise RuntimeError(f"Evaluation failed for {pending[future]}") from error
            saved[result["id"]] = result
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
            if len(saved) % 25 == 0 or len(saved) == len(rows):
                print(f"Evaluated: {len(saved)}/{len(rows)}", flush=True)
    ordered = [saved[row["id"]] for row in rows]
    result = summarize(ordered, metadata)
    write_json(output / "metrics.json", result)
    return result
