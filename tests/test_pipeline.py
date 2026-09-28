"""Offline tests for data processing, parsing, caching and graph routing."""

import json
from copy import deepcopy
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from data import LABEL_MAP, remove_overlap
from evaluate import evaluate, metrics
from graph import Retriever, cluster_rows, predict, validate_graph
from model import Model, parse_sentiment, parse_sufficiency


def edge(source, target, distance=1):
    return {"source": source, "target": target, "type": "affects", "sentiment": "Bullish", "distance": distance}


class FakeModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.prompts = []

    def chat(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return next(self.responses)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.graph = {"relations": [edge("A", "B"), edge("B", "C"), edge("C", "D"),
                                    edge("B", "E", 2), edge("a", "F")]}
        self.vectors = np.array([[1, 0], [.8, .6], [0, 1], [.6, .8], [.6, -.8]], dtype="float32")
        self.retriever = Retriever(self.graph, self.vectors)

    def test_dataset_labels(self):
        self.assertEqual(LABEL_MAP, {"negative": 0, "positive": 1, "neutral": 2})

    def test_cross_split_duplicates_are_removed(self):
        train = [{"text": "Net PROFIT  rises"}, {"text": "Sales decline"}]
        self.assertEqual(remove_overlap(train, [{"text": " net profit rises "}]), train[1:])

    def test_cluster_membership_has_one_label_and_covers_every_row(self):
        rows = [{"id": i, "label": i % 3} for i in range(12)]
        vectors = np.random.default_rng(42).normal(size=(12, 4)).astype("float32")
        groups = cluster_rows(rows, vectors, cluster_size=2, max_cluster_size=2)
        self.assertEqual(sorted(r["id"] for g in groups for r in g), list(range(12)))
        self.assertTrue(all(len({r["label"] for r in g}) == 1 and len(g) <= 2 for g in groups))

    def test_faiss_cosine_and_exactly_one_hop(self):
        vector = np.array([1, 0], dtype="float32")
        self.assertEqual(self.retriever.retrieve(vector, 1), [0])
        self.assertEqual(self.retriever.expand([0], vector, 20), [0, 1, 4])
        # C-D is two hops from A-B; B-E is an explicitly non-direct relation.
        self.assertEqual(self.retriever.expand([0], vector, 1), [0, 1])

    def test_retrieval_then_one_hop_success(self):
        model = FakeModel(["No. Missing context.", "Yes. Enough context.", "Sentiment: 1"])
        result = predict("query", self.vectors[0], self.retriever, model, top_k=1)
        self.assertEqual(result["route"], "one_hop")
        self.assertEqual(result["prediction"], 1)
        self.assertIn("B affects C", model.prompts[1])

    def test_insufficient_knowledge_falls_back_without_context(self):
        model = FakeModel(["No.", "No.", "Sentiment: [2]"])
        result = predict("query", self.vectors[0], self.retriever, model, top_k=1)
        self.assertEqual(result["route"], "direct")
        self.assertNotIn("A affects B", model.prompts[-1])
        self.assertNotIn("Knowledge:", model.prompts[-1])

    def test_sufficient_retrieval_skips_expansion(self):
        model = FakeModel(["Yes.", "Sentiment: 0"])
        with patch.object(self.retriever, "expand", side_effect=AssertionError("Unexpected expansion")):
            result = predict("query", self.vectors[0], self.retriever, model, top_k=1)
        self.assertEqual(result["route"], "retrieval")

    def test_invalid_outputs_raise_errors(self):
        self.assertEqual(parse_sentiment("Explanation\nSentiment: 2"), 2)
        self.assertFalse(parse_sufficiency("**No**. Not enough information."))
        for text in ("neutral", "Sentiment: 20", "Sentiment: [0/1/2]"):
            with self.assertRaises(ValueError):
                parse_sentiment(text)
        with self.assertRaises(ValueError):
            parse_sufficiency("Possibly yes")
        with self.assertRaises(ValueError):
            validate_graph({"entities": [], "relations": [edge("A", "B", 0)], "sentiment_label": "Bullish"}, "Bullish")

    def test_relation_endpoints_receive_default_attributes(self):
        graph = validate_graph({"entities": [], "relations": [edge("A", "B")], "sentiment_label": "Bullish"}, "Bullish")
        self.assertEqual({e["name"] for e in graph["entities"]}, {"A", "B"})
        self.assertTrue(all(e["type"] == "Unspecified" and e["attributes"] == {} for e in graph["entities"]))

    def test_descriptive_edge_sentiments_are_preserved(self):
        relation = {**edge("A", "B"), "sentiment": "Mixed"}
        graph = validate_graph({"entities": [], "relations": [relation], "sentiment_label": "Bullish"}, "Bullish")
        self.assertEqual(graph["relations"][0]["sentiment"], "Mixed")

    def test_macro_f1_uses_all_three_labels(self):
        result = metrics([{"label": 2, "kg": {"prediction": 2}}], "kg")
        self.assertAlmostEqual(result["macro_f1"], 1 / 3)
        self.assertEqual(metrics([], "kg")["accuracy"], None)

    def test_evaluation_and_resume(self):
        model = FakeModel(["Sentiment: 2", "Yes.", "Sentiment: 1"])
        model.name, model.embedding, model.seed = "test", "test", 42
        model.embed = lambda texts: np.repeat(self.vectors[:1], len(texts), axis=0)
        dataset = {"metadata": {"dataset": "example", "train_sha256": "training"},
                   "evaluation": [{"id": "q1", "text": "query", "label": 2}]}
        graph = {"metadata": {"data": {"train_sha256": "training"}},
                 "relations": [edge("A", "B")]}
        with TemporaryDirectory() as directory:
            result = evaluate(dataset, graph, model, directory, workers=1)
            self.assertEqual(set(result["metrics"]), {"cot", "kg"})
            self.assertEqual(result["metrics"]["cot"]["accuracy"], 1.0)
            self.assertEqual(result["metrics"]["kg"]["accuracy"], 0.0)
            row = json.loads(Path(directory, "predictions.jsonl").read_text())
            self.assertEqual(set(row), {"id", "text", "label", "cot", "kg"})
            self.assertEqual(evaluate(dataset, graph, model, directory, workers=1), result)
            self.assertEqual(len(model.prompts), 3)

    def test_resume_rejects_corrupted_records_before_api_calls(self):
        model = FakeModel(["Sentiment: 2", "Yes.", "Sentiment: 1"])
        model.name, model.embedding, model.seed = "test", "test", 42
        model.embed = lambda texts: np.repeat(self.vectors[:1], len(texts), axis=0)
        dataset = {"metadata": {"dataset": "example", "train_sha256": "training"},
                   "evaluation": [{"id": "q1", "text": "query", "label": 2}]}
        graph = {"metadata": {"data": {"train_sha256": "training"}},
                 "relations": [edge("A", "B")]}
        with TemporaryDirectory() as directory:
            evaluate(dataset, graph, model, directory, workers=1)
            path = Path(directory, "predictions.jsonl")
            original = path.read_text()
            row = json.loads(original)
            invalid = [original + original]
            for key, value in (("text", "different query"), ("label", 0), ("id", "unknown")):
                invalid.append(json.dumps({**row, key: value}) + "\n")
            for prediction in (2, 3, True):
                changed = deepcopy(row)
                changed["kg"]["prediction"] = prediction
                invalid.append(json.dumps(changed) + "\n")
            changed = deepcopy(row)
            changed["kg"]["route"] = "unknown"
            invalid.append(json.dumps(changed) + "\n")
            model.embed = Mock(side_effect=AssertionError("Unexpected embedding call"))
            for content in invalid:
                with self.subTest(content=content):
                    path.write_text(content)
                    with self.assertRaises(ValueError):
                        evaluate(dataset, graph, model, directory, workers=1)
            path.write_text(original)
            Path(directory, "evaluation.json").unlink()
            with self.assertRaisesRegex(ValueError, "no configuration manifest"):
                evaluate(dataset, graph, model, directory, workers=1)
            self.assertFalse(Path(directory, "evaluation.json").exists())
            self.assertEqual(len(model.prompts), 3)

    def test_truncated_response_retries_and_only_caches_complete_output(self):
        def response(reason, text):
            return SimpleNamespace(choices=[SimpleNamespace(finish_reason=reason,
                message=SimpleNamespace(content=text))], model="test", system_fingerprint="test",
                usage=SimpleNamespace(model_dump=lambda: {}))
        create = Mock(side_effect=[response("length", "incomplete"), response("stop", "Yes. Complete.")])
        client = SimpleNamespace(base_url="https://example.invalid/v1/",
                                 chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with TemporaryDirectory() as directory, patch("model.OpenAI", return_value=client):
            model = Model(cache=f"{directory}/cache.sqlite")
            self.assertEqual(model.chat("question", max_tokens=64), "Yes. Complete.")
            self.assertEqual(model.chat("question", max_tokens=64), "Yes. Complete.")
            self.assertEqual([c.kwargs["max_tokens"] for c in create.call_args_list], [64, 512])
            model.db.close()


if __name__ == "__main__":
    unittest.main()
