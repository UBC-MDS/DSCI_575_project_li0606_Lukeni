from pathlib import Path

from src.utils import load_jsonl_gz, build_corpus, save_corpus
from src.bm25 import BM25Retriever
from src.semantic import SemanticRetriever


def main() -> None:
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    reviews_path = raw_dir / "Video_Games.jsonl.gz"
    meta_path = raw_dir / "meta_Video_Games.jsonl.gz"

    reviews_df = load_jsonl_gz(reviews_path)
    meta_df = load_jsonl_gz(meta_path)

    corpus_df = build_corpus(reviews_df, meta_df)
    save_corpus(corpus_df, processed_dir / "video_games_corpus.parquet")

    bm25 = BM25Retriever(corpus_df)
    bm25.save(
        processed_dir / "bm25_index.pkl",
        processed_dir / "bm25_tokens.pkl",
    )

    semantic = SemanticRetriever(corpus_df)
    semantic.build_index()
    semantic.save(
        processed_dir / "faiss.index",
        processed_dir / "semantic_metadata.pkl",
    )

    print("Finished building corpus, BM25 index, and semantic index.")


if __name__ == "__main__":
    main()