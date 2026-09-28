"""Usage: python run.py run --dataset phrasebank100"""

import argparse
import os
from pathlib import Path

from dotenv import dotenv_values

import data
from evaluate import evaluate
from graph import build, merge_graphs
from model import Model


def main():
    parser = argparse.ArgumentParser(description="Financial sentiment analysis with KG-RAG")
    parser.add_argument("command", choices=("prepare", "build", "merge", "evaluate", "run"))
    parser.add_argument("--dataset", choices=(*data.DATASETS, "all"), default="phrasebank100")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default="runs/main")
    parser.add_argument("--sources", nargs="+", choices=data.DATASETS,
                        default=["phrasebank100", "twitter"], help="Datasets whose graphs to merge")
    parser.add_argument("--source-output", default="runs/main", help="Directory containing source graphs")
    parser.add_argument("--graph", help="Shared graph JSON for evaluation")
    parser.add_argument("--env-file", default=".env", help="Explicit local settings override inherited environment")
    parser.add_argument("--model", default="gpt-4o-mini-2024-07-18")
    parser.add_argument("--embedding", default="text-embedding-ada-002")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cluster-size", type=int, default=20)
    parser.add_argument("--max-cluster-size", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-neighbors", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, help="Stratified evaluation sample; omitted means full evaluation")
    args = parser.parse_args()
    for name in ("cluster_size", "max_cluster_size", "top_k", "workers"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_neighbors < 0 or (args.limit is not None and args.limit < 3):
        parser.error("--max-neighbors must be nonnegative and --limit at least 3")
    if args.graph and args.command != "evaluate":
        parser.error("--graph is used with evaluate")
    if args.command == "merge":
        if len(args.sources) < 2 or len(set(args.sources)) != len(args.sources):
            parser.error("--sources requires at least two distinct datasets")
        datasets = {name: data.load(name, args.data_dir, args.seed) for name in args.sources}
        graphs = {name: data.read_json(Path(args.source_output) / name / "graph.json") for name in args.sources}
        merged = merge_graphs(graphs, datasets)
        target = Path(args.output) / "graph.json"
        if target.exists() and data.read_json(target) != merged:
            raise ValueError("Merged graph changed; choose a new --output directory")
        data.write_json(target, merged)
        print(f"Merged {len(graphs)} graphs: {len(merged['relations'])} relations -> {target}", flush=True)
        return
    for key, value in dotenv_values(args.env_file).items():
        if value is not None:
            os.environ[key] = value
    shared_graph = data.read_json(args.graph) if args.graph else None
    available = (tuple(shared_graph["metadata"]["sources"])
                 if shared_graph and shared_graph["metadata"].get("kind") == "merged" else data.DATASETS)
    datasets = available if args.dataset == "all" else (args.dataset,)
    model = None if args.command == "prepare" else Model(args.model, args.embedding, seed=args.seed)
    for name in datasets:
        print(f"\nDataset: {name}", flush=True)
        dataset = data.prepare(name, args.data_dir, args.seed) if args.command == "prepare" else data.load(name, args.data_dir, args.seed)
        if args.command == "prepare":
            print(dataset["metadata"], flush=True)
            continue
        output = Path(args.output) / name
        if args.command in ("build", "run"):
            graph = build(dataset, model, output, args.cluster_size, args.max_cluster_size, args.workers)
        else:
            graph = shared_graph if shared_graph is not None else data.read_json(output / "graph.json")
        if args.command in ("evaluate", "run"):
            result = evaluate(dataset, graph, model, output, args.limit, args.workers, args.top_k, args.max_neighbors)
            for method, metric in result["metrics"].items():
                print(f"{method:6} accuracy={metric['accuracy']:.4f} macro_f1={metric['macro_f1']:.4f}", flush=True)


if __name__ == "__main__":
    main()
