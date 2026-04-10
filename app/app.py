import os

import streamlit as st
from dotenv import load_dotenv


def main() -> None:
    load_dotenv()

    st.set_page_config(page_title="DSCI 575 Retrieval Demo", layout="wide")
    st.title("Retrieval Demo")

    mode = st.radio("Search mode", ["BM25", "Semantic", "Hybrid (optional)"], horizontal=True)
    query = st.text_input("Query", placeholder="e.g., wireless bluetooth headphones")

    st.caption(
        "Connect this app to BM25/FAISS indices built in `src/` when ready."
    )

    if st.button("Search", type="primary", disabled=not query.strip()):
        st.subheader("Results")
        st.info(f"Mode: {mode}. Query: {query!r}. Connect retrieval logic in `app/app.py`.")

    with st.expander("Environment", expanded=False):
        st.write(
            {
                "AMAZON_CATEGORY": os.getenv("AMAZON_CATEGORY"),
                "RAW_DATA_DIR": os.getenv("RAW_DATA_DIR"),
                "PROCESSED_DATA_DIR": os.getenv("PROCESSED_DATA_DIR"),
            }
        )


if __name__ == "__main__":
    main()

