"""Pinned public datasets and the paper's 70/30 PhraseBank split."""

import csv
import hashlib
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

import httpx
from sklearn.model_selection import train_test_split

DATASETS = ("phrasebank50", "phrasebank100", "twitter")
PB_REV = "8d3fe0c36d5feec6b3cc5e455b0fcb4820fb9964"
TW_REV = "ccbe24de388e287beb92dd393a335c376b350ac3"
LABEL_MAP = {"negative": 0, "positive": 1, "neutral": 2}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def normalized(text):
    return " ".join(text.casefold().split())


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def download(path, repo, revision, filename):
    path = Path(path)
    if not path.exists():
        response = httpx.get(f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{filename}",
                             follow_redirects=True, timeout=120)
        response.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
    return path.read_bytes()


def remove_overlap(train, evaluation):
    """Remove training texts matching normalized evaluation texts."""
    heldout = {normalized(row["text"]) for row in evaluation}
    return [row for row in train if normalized(row["text"]) not in heldout]


def prepare(name, root="data", seed=42):
    root = Path(root)
    sources = []
    if name.startswith("phrasebank"):
        raw = download(root / "raw/FinancialPhraseBank-v1.0.zip", "takala/financial_phrasebank",
                       PB_REV, "data/FinancialPhraseBank-v1.0.zip")
        variant = "50Agree" if name == "phrasebank50" else "AllAgree"
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            lines = archive.read(f"FinancialPhraseBank-v1.0/Sentences_{variant}.txt").decode("latin-1").splitlines()
        rows = []
        for i, line in enumerate(lines):
            text, label = line.rsplit("@", 1)
            rows.append({"id": f"{name}:{i}", "text": text.strip(), "label": LABEL_MAP[label.strip()]})
        train, evaluation = train_test_split(rows, test_size=0.3, random_state=seed)
        sources.append({"repo": "takala/financial_phrasebank", "revision": PB_REV,
                        "sha256": hashlib.sha256(raw).hexdigest()})
    elif name == "twitter":
        splits = []
        for split in ("train", "valid"):
            raw = download(root / f"raw/sent_{split}.csv", "zeroshot/twitter-financial-news-sentiment",
                           TW_REV, f"sent_{split}.csv")
            rows = [{"id": f"twitter:{split}:{i}", "text": row["text"].strip(), "label": int(row["label"])}
                    for i, row in enumerate(csv.DictReader(io.StringIO(raw.decode("utf-8"))))]
            if any(row["label"] not in (0, 1, 2) for row in rows):
                raise ValueError("Unexpected Twitter label")
            splits.append(rows)
            sources.append({"repo": "zeroshot/twitter-financial-news-sentiment", "revision": TW_REV,
                            "file": f"sent_{split}.csv", "sha256": hashlib.sha256(raw).hexdigest()})
        train, evaluation = splits
    else:
        raise ValueError(f"Unknown dataset: {name}")
    clean_train = remove_overlap(train, evaluation)
    metadata = {"dataset": name, "seed": seed, "sources": sources,
                "label_mapping": {0: "Bearish", 1: "Bullish", 2: "Neutral"},
                "original_train_count": len(train), "train_count": len(clean_train),
                "eval_count": len(evaluation), "excluded_overlap_rows": len(train) - len(clean_train),
                "original_train_labels": dict(Counter(r["label"] for r in train)),
                "eval_labels": dict(Counter(r["label"] for r in evaluation)),
                "train_sha256": digest(clean_train), "eval_sha256": digest(evaluation)}
    result = {"metadata": metadata, "train": clean_train, "evaluation": evaluation}
    write_json(root / f"{name}.json", result)
    return result


def load(name, root="data", seed=42):
    path = Path(root) / f"{name}.json"
    if not path.exists():
        return prepare(name, root, seed)
    result = read_json(path)
    if result["metadata"]["seed"] != seed:
        raise ValueError("Prepared split has a different seed; run prepare again or use another --data-dir")
    for key, split in (("train_sha256", "train"), ("eval_sha256", "evaluation")):
        if digest(result[split]) != result["metadata"][key]:
            raise ValueError(f"Prepared {split} data do not match their checksum")
    if len(remove_overlap(result["train"], result["evaluation"])) != len(result["train"]):
        raise ValueError("Training/evaluation text overlap")
    return result
