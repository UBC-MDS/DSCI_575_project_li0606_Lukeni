from pathlib import Path

from src.utils import load_jsonl, build_corpus, save_corpus
from src.bm25 import BM25Retriever
from src.semantic import SemanticRetriever

MIN_PRODUCTS = 10000
MAX_REVIEWS_PER_PRODUCT = 20  # set to an integer like 20 only if runtime is too slow


def main() -> None:
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    reviews_path = raw_dir / "Video_Games.jsonl"
    meta_path = raw_dir / "meta_Video_Games.jsonl"

    reviews_df = load_jsonl(reviews_path)
    meta_df = load_jsonl(meta_path)

    join_key = "parent_asin" if "parent_asin" in reviews_df.columns and "parent_asin" in meta_df.columns else "asin"

    # choose at least 10,000 unique products
    selected_products = (
        reviews_df[join_key]
        .dropna()
        .drop_duplicates()
        .iloc[:MIN_PRODUCTS]
    )

    reviews_df = reviews_df[reviews_df[join_key].isin(selected_products)].copy()
    meta_df = meta_df[meta_df[join_key].isin(selected_products)].copy()

    # optional runtime control
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
    save_corpus(corpus_df, processed_dir / "video_games_corpus.parquet")

    bm25 = BM25Retriever(corpus_df)
    bm25.save(
        processed_dir / "bm25_index.pkl",
        processed_dir / "bm25_tokens.pkl",
    )

    semantic = SemanticRetriever(corpus_df)
    semantic.build_index(batch_size=16)
    semantic.save(
        processed_dir / "faiss.index",
        processed_dir / "semantic_metadata.pkl",
    )

    print("Finished building corpus, BM25 index, and semantic index.")

if __name__ == "__main__":
    main()