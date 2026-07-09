import argparse
import json
from pathlib import Path

from rag_service.embedder import DEFAULT_EMBEDDING_MODEL, SentenceTransformerEmbedder
from rag_service.pipeline import ask_question, build_index

EXAMPLES_DIR = Path("examples")
EVAL_FILE = EXAMPLES_DIR / "eval_questions.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval hit@k on example documents.")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--min-score", type=float, default=0.25)
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    args = parser.parse_args()

    eval_items = json.loads(EVAL_FILE.read_text(encoding="utf-8"))
    document_paths = [
        path
        for path in EXAMPLES_DIR.iterdir()
        if path.suffix.lower() in {".txt", ".md", ".pdf", ".docx", ".html", ".ipynb"}
    ]
    embedder = SentenceTransformerEmbedder(args.model)
    store = build_index(document_paths, embedder)

    hits = 0
    for item in eval_items:
        _answer, results = ask_question(
            item["question"],
            store,
            embedder,
            top_k=args.top_k,
            min_score=args.min_score,
        )
        sources = {chunk.metadata.get("source") for chunk, _score in results}
        expected = set(item["expected_sources"])
        hit = bool(sources & expected)
        hits += int(hit)
        status = "hit" if hit else "miss"
        print(f"{status}: {item['question']} -> {sorted(sources)}")

    score = hits / len(eval_items) if eval_items else 0
    print(f"hit@{args.top_k}: {hits}/{len(eval_items)} = {score:.2f}")


if __name__ == "__main__":
    main()
