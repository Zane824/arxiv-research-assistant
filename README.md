# arXiv Research Assistant

A retrieval-augmented question-answering system over recent arXiv papers, built around
**measurement**: an evaluation harness scores each retriever on a labeled question set with
Recall@k and MRR, so retrieval choices are compared rather than judged by eye.

- **Ingestion** from the public arXiv API (rate-limited, deduplicated, resumable)
- **Retrieval**: BM25, dense embeddings (BGE), and hybrid reciprocal rank fusion
- **Answering** with an LLM that must cite `[arxiv_id]` for every claim; citations that point
  outside the retrieved context are flagged as unsupported
- **Evaluation**: Recall@k and MRR per retriever on a labeled question set
- Runs fully locally (Ollama), or against any OpenAI-compatible endpoint

## Architecture

```mermaid
flowchart LR
    A[arXiv API] -->|ingest| B[(papers.jsonl)]
    B --> C[BM25 index]
    B --> D[Dense index<br/>bge-small]
    Q[Question] --> C & D
    C & D --> E[Reciprocal rank fusion]
    E --> F[Reranker<br/><i>planned</i>]
    F --> G[LLM answer<br/>with citations]
    E --> G
    H[(Labeled eval set)] --> I[Recall@k / MRR]
    E --> I
```

## Results

`arxiv-rag eval` scores every retriever on the same labeled question set and reports Recall@1/5/10
and MRR, so BM25, dense and hybrid are compared on identical ground.

Numbers are not published yet: the labeled question set is still being written, and figures
produced from the unlabeled template would not measure anything. The table lands with the first
full evaluation run — see the Roadmap below.

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows; on macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"            # BM25 only
pip install -e ".[dense]"          # + dense retrieval (installs PyTorch)

arxiv-rag ingest --category cs.CL cs.IR --max 2000
arxiv-rag search "retrieval augmented generation for long documents" --retriever bm25
arxiv-rag eval-template -n 60      # then write a question for each sampled paper
arxiv-rag eval                     # compare retrievers
```

Answering needs an LLM server. With [Ollama](https://ollama.com):

```bash
ollama pull qwen2.5:7b
arxiv-rag ask "How do recent papers reduce hallucination in RAG?" --retriever hybrid
```

Use another OpenAI-compatible endpoint by setting `ARXIV_RAG_LLM_BASE_URL`, `ARXIV_RAG_LLM_MODEL`
and `ARXIV_RAG_LLM_API_KEY` (see `.env.example`).

### Docker

```bash
docker build -t arxiv-rag .                          # add --build-arg EXTRAS=dense for embeddings
docker run --rm -v "$PWD/data:/data" arxiv-rag ingest --max 500
docker run --rm -v "$PWD/data:/data" arxiv-rag search "speculative decoding"
```

From inside the container, a host Ollama server is at `http://host.docker.internal:11434/v1`.

## Evaluation set

`data/eval/questions.jsonl`, one JSON object per line:

```json
{"qid": "q001", "question": "Which method verifies draft tokens in parallel?", "relevant_ids": ["2401.00003"]}
```

`arxiv-rag eval-template` samples papers so each question has a known relevant paper. Questions
are written by hand, paraphrasing rather than copying the abstract, so that lexical overlap does
not trivially favor BM25.

## Roadmap

- [x] arXiv ingestion, BM25 / dense / hybrid retrieval, retrieval evaluation, cited answers
- [ ] Labeled eval set (60+ questions) and first results table
- [ ] Cross-encoder reranking (`src/arxiv_rag/rerank.py`)
- [ ] Multi-step agent in LangGraph: retrieve, grade, rewrite query (`src/arxiv_rag/agent.py`)
- [ ] Answer-level evaluation: citation precision, unsupported-citation rate
- [ ] Simple web UI (Streamlit or Gradio)

## Project layout

```
src/arxiv_rag/
  ingest.py      arXiv API client and Atom parser
  corpus.py      Paper record, JSONL storage
  retrieval.py   BM25, dense, hybrid (RRF)
  rerank.py      cross-encoder reranker (planned)
  rag.py         prompt, citation extraction, single-pass answering
  agent.py       multi-step agent (planned)
  evaluation.py  Recall@k, MRR, eval-set tooling
  llm.py         OpenAI-compatible chat client
  cli.py         `arxiv-rag` command
tests/           pytest suite (no network or GPU needed)
```
