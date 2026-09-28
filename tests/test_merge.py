"""Merged-graph provenance, split isolation and shared-graph evaluation."""

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import numpy as np

from data import digest, read_json, write_json
from evaluate import evaluate
from graph import Retriever, merge_graphs
import run
from test_pipeline import FakeModel, edge


class MergeTests(unittest.TestCase):
    def setUp(self):
        self.data, self.graphs = {}, {}
        for name in ("phrasebank100", "twitter"):
            train = [{"id": name + ":train", "text": name + " training", "label": 1}]
            evaluation = [{"id": name + ":test", "text": name + " query", "label": 0}]
            metadata = {"dataset": name, "seed": 42,
                        "train_sha256": digest(train), "eval_sha256": digest(evaluation)}
            self.data[name] = {"metadata": metadata, "train": train, "evaluation": evaluation}
            self.graphs[name] = {
                "metadata": {"data": deepcopy(metadata)},
                "entities": [{"name": "A", "type": "Company", "attributes": {}, "group_id": 0}],
                "relations": [{**edge("A", "B"), "group_ids": [0]}]}
        self.graphs["twitter"]["relations"] = [
            {**edge("a", "b"), "group_ids": [0]},
            {**edge("B", "C"), "group_ids": [0]},
            {**edge("A", "B"), "sentiment": "Bearish", "group_ids": [0]}]

    def test_union_keeps_origins_conflicting_sentiments_and_cross_graph_neighbors(self):
        original = deepcopy(self.graphs)
        merged = merge_graphs(self.graphs, self.data)
        self.assertEqual(self.graphs, original)
        self.assertEqual(len(merged["relations"]), 3)
        self.assertEqual(merged["relations"][0]["group_ids"], ["phrasebank100:0", "twitter:0"])
        self.assertEqual({e["group_id"] for e in merged["entities"]}, {"phrasebank100:0", "twitter:0"})
        self.assertEqual(merged["relations"][2]["sentiment"], "Bearish")
        vectors = np.array([[1, 0], [.8, .6], [0, 1]], dtype="float32")
        retriever = Retriever(merged, vectors)
        self.assertEqual(retriever.expand([0], vectors[0], 1), [0, 1])
        self.assertEqual(merged, merge_graphs(dict(reversed(list(self.graphs.items()))), self.data))

    def test_cross_dataset_heldout_overlap_is_rejected(self):
        source = self.data["twitter"]
        source["train"][0]["text"] = self.data["phrasebank100"]["evaluation"][0]["text"].upper()
        source["metadata"]["train_sha256"] = digest(source["train"])
        self.graphs["twitter"]["metadata"]["data"] = deepcopy(source["metadata"])
        with self.assertRaisesRegex(ValueError, "held-out"):
            merge_graphs(self.graphs, self.data)

    def test_stale_source_graph_and_changed_data_are_rejected(self):
        changed = deepcopy(self.graphs)
        changed["twitter"]["metadata"]["data"]["train_sha256"] = "old"
        with self.assertRaisesRegex(ValueError, "does not match"):
            merge_graphs(changed, self.data)
        self.data["twitter"]["train"][0]["text"] = "changed"
        with self.assertRaisesRegex(ValueError, "checksum"):
            merge_graphs(self.graphs, self.data)

    def test_evaluation_rejects_unknown_or_changed_source_split(self):
        merged = merge_graphs(self.graphs, self.data)
        with TemporaryDirectory() as directory:
            for key, value in (("dataset", "unknown"), ("eval_sha256", "changed")):
                changed = deepcopy(self.data["twitter"])
                changed["metadata"][key] = value
                with self.assertRaisesRegex(ValueError, "matching merged-graph source"):
                    evaluate(changed, merged, Mock(), directory)

    def test_cli_merge_and_evaluate_all_sources(self):
        model = FakeModel(["Sentiment: 0", "Yes.", "Sentiment: 1"] * 2)
        model.name, model.embedding, model.seed = "test", "test", 42
        model.embed = lambda texts: np.tile(np.array([[1, 0]], dtype="float32"), (len(texts), 1))
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for name in self.data:
                write_json(root / "data" / f"{name}.json", self.data[name])
                write_json(root / "sources" / name / "graph.json", self.graphs[name])
            common = ["--data-dir", str(root / "data"), "--output", str(root / "merged")]
            argv = ["run.py", "merge", "--source-output", str(root / "sources"), *common]
            with patch("sys.argv", argv), patch("run.Model", side_effect=AssertionError("Unexpected API client")):
                run.main()
            argv = ["run.py", "evaluate", "--dataset", "all", "--graph",
                    str(root / "merged/graph.json"), "--workers", "1", *common]
            with patch("sys.argv", argv), patch("run.dotenv_values", return_value={}), patch("run.Model", return_value=model):
                run.main()
            for name in self.data:
                metrics = read_json(root / "merged" / name / "metrics.json")
                self.assertEqual(metrics["metrics"]["cot"]["accuracy"], 1.0)
                self.assertEqual(metrics["metrics"]["kg"]["accuracy"], 0.0)
            self.assertFalse((root / "merged/phrasebank50").exists())
            self.assertEqual(len(model.prompts), 6)


if __name__ == "__main__":
    unittest.main()
