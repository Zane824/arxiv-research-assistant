"""Command line entry point: `arxiv-rag <command> --help`."""

from __future__ import annotations

import argparse
import json
import sys

import httpx

from .config import Settings
from .corpus import Paper, load_papers, merge_papers, save_papers
from .evaluation import evaluate_retriever, format_table, load_eval_set, make_template
from .ingest import fetch_category
from .retrieval import BM25Retriever, DenseRetriever, HybridRetriever, Retriever

RETRIEVERS = ("bm25", "dense", "hybrid")


def build_retriever(name: str, papers: list[Paper], settings: Settings) -> Retriever:
    if name == "bm25":
        return BM25Retriever(papers)
    if name == "dense":
        return DenseRetriever(papers, settings.embedding_model, cache_dir=settings.index_dir)
    if name == "hybrid":
        return HybridRetriever(
            [build_retriever("bm25", papers, settings), build_retriever("dense", papers, settings)]
        )
    raise ValueError(f"Unknown retriever: {name}")


def _require_corpus(settings: Settings) -> list[Paper]:
    papers = load_papers(settings.corpus_path)
    if not papers:
        sys.exit(f"No corpus at {settings.corpus_path}. Run `arxiv-rag ingest` first.")
    return papers


def cmd_ingest(args: argparse.Namespace, settings: Settings) -> None:
    existing = load_papers(settings.corpus_path)
    fetched: list[Paper] = []
    error = None
    for category in args.category:
        print(f"Fetching up to {args.max} papers from {category} (offset {args.start}) ...")
        before = len(fetched)
        try:
            for paper in fetch_category(category, args.max, start=args.start):
                fetched.append(paper)
        except httpx.HTTPError as exc:
            # Keep what we already have; rerun with --start to continue this category.
            error = f"{category}: {exc} (got {len(fetched) - before}; resume with --start)"
            break
    merged = merge_papers(existing, fetched)
    save_papers(settings.corpus_path, merged)
    print(f"Fetched {len(fetched)}; corpus now has {len(merged)} papers -> {settings.corpus_path}")
    if error:
        sys.exit(f"Stopped early. {error}")


def cmd_search(args: argparse.Namespace, settings: Settings) -> None:
    retriever = build_retriever(args.retriever, _require_corpus(settings), settings)
    for rank, hit in enumerate(retriever.search(args.query, args.k), start=1):
        p = hit.paper
        print(f"{rank:2d}. [{p.arxiv_id}] {p.title}  ({p.published}, score={hit.score:.3f})")


def cmd_ask(args: argparse.Namespace, settings: Settings) -> None:
    from .llm import OpenAICompatibleChat
    from .rag import answer_question

    retriever = build_retriever(args.retriever, _require_corpus(settings), settings)
    llm = OpenAICompatibleChat(settings.llm_base_url, settings.llm_model, settings.llm_api_key)
    answer = answer_question(args.question, retriever, llm, args.k)
    print(answer.text)
    print("\nRetrieved:")
    for hit in answer.hits:
        print(f"  [{hit.paper.arxiv_id}] {hit.paper.title}")
    if answer.unsupported_citations:
        print(f"\nWarning: cited ids not in context: {', '.join(answer.unsupported_citations)}")


def cmd_eval_template(args: argparse.Namespace, settings: Settings) -> None:
    out = settings.eval_path.with_name("questions_template.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = make_template(_require_corpus(settings), args.n, args.seed)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} rows to {out}.")
    print(f"Fill in each 'question', then save the file as {settings.eval_path}")


def cmd_eval(args: argparse.Namespace, settings: Settings) -> None:
    papers = _require_corpus(settings)
    if not settings.eval_path.exists():
        sys.exit(f"No eval set at {settings.eval_path}. Run `arxiv-rag eval-template`, label it.")
    items = load_eval_set(settings.eval_path)
    ks = tuple(int(k) for k in args.k.split(","))
    names = RETRIEVERS if args.retriever == "all" else (args.retriever,)
    results = {}
    for name in names:
        try:
            retriever = build_retriever(name, papers, settings)
        except ImportError as exc:
            print(f"Skipping {name}: {exc}")
            continue
        results[name] = evaluate_retriever(retriever, items, ks)
    if not results:
        sys.exit("No retriever could be evaluated.")
    table = format_table(results, len(items))
    print(table)
    settings.results_dir.mkdir(parents=True, exist_ok=True)
    out = settings.results_dir / "retrieval_eval.md"
    out.write_text(table + "\n", encoding="utf-8")
    print(f"\nSaved to {out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="arxiv-rag", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="Download paper metadata from the arXiv API")
    p.add_argument("--category", nargs="+", default=["cs.CL"], help="e.g. cs.CL cs.LG cs.IR")
    p.add_argument("--max", type=int, default=1000, help="Papers per category (most recent first)")
    p.add_argument("--start", type=int, default=0, help="Skip this many of the most recent papers")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("search", help="Show top-k retrieved papers for a query")
    p.add_argument("query")
    p.add_argument("--retriever", choices=RETRIEVERS, default="bm25")
    p.add_argument("-k", type=int, default=10)
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("ask", help="Answer a question with citations (needs an LLM server)")
    p.add_argument("question")
    p.add_argument("--retriever", choices=RETRIEVERS, default="bm25")
    p.add_argument("-k", type=int, default=5)
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("eval-template", help="Sample papers to write eval questions for")
    p.add_argument("-n", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_eval_template)

    p = sub.add_parser("eval", help="Recall@k / MRR of each retriever on the labeled set")
    p.add_argument("--retriever", choices=(*RETRIEVERS, "all"), default="all")
    p.add_argument("-k", default="1,5,10", help="Comma-separated cutoffs")
    p.set_defaults(func=cmd_eval)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args, Settings())


if __name__ == "__main__":
    main()
