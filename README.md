# Financial Sentiment Analysis with KG-RAG

Implementation of the method described in **Combining LLM-Generated Knowledge
Graphs with RAG for Financial Sentiment Extraction**, by Zihao Huang, Kelvin Du,
Xulang Zhang, Rui Mao, and Erik Cambria.

Paper: [PDF](https://sentic.net/knowledge-graphs-with-rag-for-financial-sentiment-extraction.pdf).

The method combines LLM-generated knowledge graphs with retrieval-augmented
generation for financial sentiment classification. See the paper for experimental
results and analysis.

## Method

1. **Cluster texts.** Embed the training texts and apply K-Means separately within
   each sentiment label.
2. **Build the graph.** Merge each cluster into a coherent text, then extract
   entities and sentiment-labelled relations. The extraction prompt covers
   numerical, temporal, comparative, causal, and risk information.
3. **Retrieve knowledge.** Convert relations into sentences, embed them with
   `text-embedding-ada-002`, and retrieve relevant relations using FAISS cosine
   similarity.
4. **Expand context.** Check whether the retrieved knowledge is sufficient. When
   needed, retrieve one-hop neighbours and check again.
5. **Predict sentiment.** Use the retrieved context to classify the query. If the
   evidence remains insufficient, let the LLM classify the query directly.

Labels: **0 = Bearish/Negative, 1 = Bullish/Positive, 2 = Neutral**.

## Installation

Requires Python 3.12+.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp -n .env.example .env
```

Set `OPENAI_API_KEY` in `.env`. An optional `OPENAI_BASE_URL` selects the API
endpoint. Settings in `.env` take precedence over the shell environment.

## Usage

Run the full pipeline:

```bash
python run.py run --dataset phrasebank100
```

Run individual stages or select another dataset:

```bash
python run.py prepare --dataset all
python run.py build --dataset phrasebank100
python run.py evaluate --dataset phrasebank100
python run.py run --dataset all --workers 16
```

For a smaller evaluation, select a stratified sample with `--limit`:

```bash
python run.py run --dataset phrasebank100 --limit 100 --output runs/sample
```

API responses and completed examples are cached. Repeat the same command to
resume a run. Use a separate `--output` directory for each configuration.

### Merged knowledge graph

Build the PhraseBank 100% and Twitter graphs, merge them, then evaluate both
datasets with the shared graph:

```bash
python run.py build --dataset phrasebank100
python run.py build --dataset twitter
python run.py merge --output runs/merged
python run.py evaluate --dataset all --graph runs/merged/graph.json --output runs/merged
```

`merge` reads graphs from `--source-output` (default: `runs/main`). Use `--sources`
to select the source datasets. Matching relations are deduplicated, source cluster
IDs are retained, and shared entity names connect the graphs during one-hop
retrieval. Training texts are checked against every source dataset's evaluation
split before merging. With a merged `--graph`, `--dataset all` evaluates its
source datasets. Results are saved separately under `runs/merged/<dataset>/`.

## Data and configuration

Datasets are downloaded automatically from
[Financial PhraseBank](https://huggingface.co/datasets/takala/financial_phrasebank)
and [Twitter Financial News](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment).
Available dataset names are `phrasebank50`, `phrasebank100`, and `twitter`.
PhraseBank uses a 70/30 split with seed 42; Twitter uses its supplied train and
validation splits. Training texts duplicated in the evaluation split are removed
before graph construction.

| Argument | Default | Purpose |
| --- | --- | --- |
| `--model` | `gpt-4o-mini-2024-07-18` | Text merging, graph extraction and classification |
| `--embedding` | `text-embedding-ada-002` | Clustering and retrieval embeddings |
| `--cluster-size` | `20` | Target texts per cluster |
| `--max-cluster-size` | `30` | Maximum texts per merge |
| `--top-k` | `5` | Initial retrieved relations |
| `--max-neighbors` | `20` | Additional one-hop relations |
| `--seed` | `42` | Data split, clustering and generation seed |
| `--workers` | `8` | Concurrent processing workers |
| `--output` | `runs/main` | Output directory |

K-Means uses `ceil(n / cluster_size)` clusters within each label, with
`n_init=10`. Oversized clusters are split into groups of at most
`max_cluster_size` texts. Generation uses temperature 0.

## Outputs

Each dataset has its own directory under `runs/main/`:

- `groups/`: merged texts, source IDs and extracted graphs.
- `graph.json`: combined graph with entity attributes and sentiment-labelled relations.
- `evaluation.json`: configuration and data checksums.
- `predictions.jsonl`: CoT and KG-RAG predictions, retrieved relation IDs and routing decisions.
- `metrics.json`: accuracy, macro-F1, confusion matrices and retrieval/fallback breakdown.

## Code

| File | Description |
| --- | --- |
| `data.py` | Dataset loading, label mapping and splitting |
| `prompts.py` | Text merging, graph extraction, sufficiency and classification prompts |
| `model.py` | API calls, response parsing and caching |
| `graph.py` | Clustering, graph construction and merging, retrieval and one-hop expansion |
| `evaluate.py` | CoT and KG-RAG evaluation |
| `run.py` | Command-line entry point |

Run the offline tests with:

```bash
python -m unittest discover -s tests -v
```

PhraseBank is distributed under CC BY-NC-SA 3.0; Twitter Financial News uses the
MIT licence.

## Citation

If you find this work useful, please consider citing our paper:

```bibtex
@INPROCEEDINGS{11415819,
  author={Huang, Zihao and Du, Kelvin and Zhang, Xulang and Mao, Rui and Cambria, Erik},
  booktitle={2025 IEEE International Conference on Data Mining Workshops (ICDMW)},
  title={Combining LLM-Generated Knowledge Graphs with RAG for Financial Sentiment Extraction},
  year={2025},
  volume={},
  number={},
  pages={2056-2063},
  keywords={Training;Sentiment analysis;Technological innovation;Accuracy;Social networking (online);Computational modeling;Retrieval augmented generation;Knowledge graphs;Market research;Reliability;financial sentiment analysis;NLP;RAG;LLM},
  doi={10.1109/ICDMW69685.2025.00250}
}
```
