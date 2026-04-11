import json
from pathlib import Path

from src.utils import load_jsonl, build_corpus, save_corpus
from src.bm25 import BM25Retriever
from src.semantic import SemanticRetriever


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main() -> None:
    base = Path("tests/tmp_test_data")
    raw_dir = base / "raw"
    processed_dir = base / "processed"

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    reviews = [
        {
            "parent_asin": "A1",
            "title": "Great racing fun",
            "text": "This racing game has fast cars and fun tracks for kids.",
            "rating": 5,
        },
        {
            "parent_asin": "A2",
            "title": "Scary adventure",
            "text": "A spooky horror game with strong story and dark atmosphere.",
            "rating": 4,
        },
        {
            "parent_asin": "A3",
            "title": "Sports challenge",
            "text": "A football sports game with smooth gameplay and realistic teams.",
            "rating": 4,
        },
    ]

    meta = [
        {
            "parent_asin": "A1",
            "title": "Turbo Racing 2024",
            "description": ["Arcade racing game", "family friendly"],
            "features": ["cars", "tracks"],
            "categories": ["Video Games", "Racing"],
            "price": 29.99,
        },
        {
            "parent_asin": "A2",
            "title": "Haunted Mansion",
            "description": ["survival horror", "story rich"],
            "features": ["dark", "puzzle"],
            "categories": ["Video Games", "Horror"],
            "price": 39.99,
        },
        {
            "parent_asin": "A3",
            "title": "Pro Football League",
            "description": ["sports simulation"],
            "features": ["teams", "career mode"],
            "categories": ["Video Games", "Sports"],
            "price": 49.99,
        },
    ]

    reviews_path = raw_dir / "Video_Games.jsonl"
    meta_path = raw_dir / "meta_Video_Games.jsonl"

    write_jsonl(reviews_path, reviews)
    write_jsonl(meta_path, meta)

    reviews_df = load_jsonl(reviews_path)
    meta_df = load_jsonl(meta_path)

    corpus_df = build_corpus(reviews_df, meta_df)
    save_corpus(corpus_df, processed_dir / "video_games_corpus.csv")

    bm25 = BM25Retriever(corpus_df)
    bm25.save(
        processed_dir / "bm25_index.pkl",
        processed_dir / "bm25_tokens.pkl",
    )
    bm25_results = bm25.search("racing cars tracks", top_k=2)
    print("\nBM25 results:")
    print(bm25_results[["product_title", "bm25_score"]])

    semantic = SemanticRetriever(corpus_df)
    semantic.build_index()
    semantic.save(
        processed_dir / "faiss.index",
        processed_dir / "semantic_metadata.pkl",
    )
    semantic_results = semantic.search("scary story game", top_k=2)
    print("\nSemantic results:")
    print(semantic_results[["product_title", "semantic_score"]])

    print("\nSmoke test finished successfully.")


if __name__ == "__main__":
    main()