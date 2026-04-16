## Model choice

For Milestone 2, we use a hosted API-based LLM instead of a local model. We selected Groq with the `llama3-8b-8192` chat model because it avoids local GPU requirements, reduces setup complexity, and provides fast response time for iterative RAG testing on a laptop.

This choice is suitable for our project because:
- we do not need to download large local model weights
- the hosted API is simpler to integrate into a retrieval pipeline
- it allows us to focus on prompt design, retrieval quality, and answer grounding