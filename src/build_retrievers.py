from src.utils import load_jsonl, build_corpus, save_corpus
from src.bm25 import BM25Retriever
from src.semantic import SemanticRetriever
from src.config import (
    PROCESSED_DIR,
    REVIEWS_PATH,
    META_PATH,
    PRODUCT_TARGET,
    MAX_REVIEWS_PER_PRODUCT,
    CORPUS_PATH,
    BM25_INDEX_PATH,
    BM25_TOKENS_PATH,
    FAISS_INDEX_PATH,
    SEMANTIC_METADATA_PATH,
)


def main() -> None:
    """Build the final scaled corpus, BM25 index, and semantic index."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    reviews_df = load_jsonl(REVIEWS_PATH)
    meta_df = load_jsonl(META_PATH)

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

    save_corpus(corpus_df, CORPUS_PATH)
    print(f"Saved final corpus to {CORPUS_PATH}")

    bm25 = BM25Retriever(corpus_df)
    bm25.save(BM25_INDEX_PATH, BM25_TOKENS_PATH)
    print(f"Saved BM25 index to {BM25_INDEX_PATH}")
    print(f"Saved BM25 tokens to {BM25_TOKENS_PATH}")

    semantic = SemanticRetriever(corpus_df)
    semantic.build_index(batch_size=4)
    print("Semantic FAISS index built.")

    semantic.save(FAISS_INDEX_PATH, SEMANTIC_METADATA_PATH)
    print(f"Saved semantic index to {FAISS_INDEX_PATH}")
    print(f"Saved semantic metadata to {SEMANTIC_METADATA_PATH}")

    print("Finished building final corpus, BM25 index, and semantic index.")


if __name__ == "__main__":
    main()