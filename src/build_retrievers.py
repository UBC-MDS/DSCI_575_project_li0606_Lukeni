from pathlib import Path

from src.utils import load_jsonl, build_corpus, save_corpus
from src.bm25 import BM25Retriever
from src.semantic import SemanticRetriever

PRODUCT_TARGET = 10000
MAX_REVIEWS_PER_PRODUCT = 3


def main() -> None:
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    reviews_path = raw_dir / "Video_Games.jsonl"
    meta_path = raw_dir / "meta_Video_Games.jsonl"

    reviews_df = load_jsonl(reviews_path)
    meta_df = load_jsonl(meta_path)

    join_key = "parent_asin" if "parent_asin" in reviews_df.columns and "parent_asin" in meta_df.columns else "asin"

    selected_products = (
        reviews_df[join_key]
        .dropna()
        .drop_duplicates()
        .iloc[:PRODUCT_TARGET]
    )

    reviews_df = reviews_df[reviews_df[join_key].isin(selected_products)].copy()
    meta_df = meta_df[meta_df[join_key].isin(selected_products)].copy()

    if MAX_REVIEWS_PER_PRODUCT is not None:
        reviews_df = (
            reviews_df.groupby(join_key, group_keys=False)
            .head(MAX_REVIEWS_PER_PRODUCT)
            .copy()
        )

    print("Selected unique products:", reviews_df[join_key].nunique())
    print("Filtered reviews shape:", reviews_df.shape)
    print("Filtered metadata shape:", meta_df.shape)

    corpus_df = build_corpus(reviews_df, meta_df)

    print("Corpus shape after build:", corpus_df.shape)
    print("Unique products after corpus build:", corpus_df[join_key].nunique())

    save_corpus(corpus_df, processed_dir / "video_games_corpus_final.parquet")
    print("Saved final corpus to data/processed/video_games_corpus_final.parquet")

    bm25 = BM25Retriever(corpus_df)
    bm25.save(
        processed_dir / "bm25_final_index.pkl",
        processed_dir / "bm25_final_tokens.pkl",
    )

    print("Saved BM25 final artifacts:")
    print("- data/processed/bm25_final_index.pkl")
    print("- data/processed/bm25_final_tokens.pkl")

    semantic = SemanticRetriever(corpus_df)
    semantic.build_index(batch_size=4)
    print("Semantic FAISS index built.")

    semantic.save(
        processed_dir / "faiss_final.index",
        processed_dir / "semantic_final_metadata.pkl",
    )

    print("Saved semantic final artifacts:")
    print("- data/processed/faiss_final.index")
    print("- data/processed/semantic_final_metadata.pkl")

    print("Finished building final corpus, BM25 index, and semantic index.")


if __name__ == "__main__":
    main()