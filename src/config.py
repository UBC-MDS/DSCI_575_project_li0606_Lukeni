from pathlib import Path
import os

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

REVIEWS_PATH = RAW_DIR / "Video_Games.jsonl"
META_PATH = RAW_DIR / "meta_Video_Games.jsonl"

PRODUCT_TARGET = 10000
MAX_REVIEWS_PER_PRODUCT = 3

CORPUS_PATH = PROCESSED_DIR / "video_games_corpus_final.parquet"
BM25_INDEX_PATH = PROCESSED_DIR / "bm25_final_index.pkl"
BM25_TOKENS_PATH = PROCESSED_DIR / "bm25_final_tokens.pkl"
FAISS_INDEX_PATH = PROCESSED_DIR / "faiss_final.index"
SEMANTIC_METADATA_PATH = PROCESSED_DIR / "semantic_final_metadata.pkl"

DEFAULT_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")