# Final Discussion

## Step 1: Improve Your Workflow

### Dataset Scaling
- Number of products used: 10,000 unique products
- Changes to sampling strategy:
  - In the earlier workflow, we worked with a smaller representative sample.
  - For the final workflow, we changed the selection logic to scale by **unique products** instead of taking an arbitrary number of rows.
  - We selected 10,000 unique products and capped the number of reviews per product at 3 to keep the build practical on local hardware.
  - After filtering, the corpus used for retrieval contained 29,589 rows and still preserved 10,000 unique products.

- Notes on the final scaled workflow:
  - BM25 and semantic retrieval were both rebuilt on the same scaled corpus.
  - This ensured the final retrieval pipeline satisfied the minimum requirement of processing at least 10,000 products.
  - The final artifacts were saved using the final naming convention rather than the old sample naming convention.

### LLM Experiment

- Models compared:
  - Primary / original model: `llama-3.1-8b-instant`
  - Comparison model: `llama-3.3-70b-versatile`

- Prompt used:

```text
You are a careful product recommendation assistant.
Use only the retrieved Amazon review context below.
If the context is insufficient, say so clearly.
Keep the answer concise, helpful, and grounded in the retrieved evidence.
```

- Results and discussion:

We compared the two models on five product-search queries while keeping the retrieved context and prompt fixed. This allowed us to compare model behavior fairly without changing the retrieval side of the pipeline.

For the query **“best racing game with fun tracks”**, both models recommended *LittleBigPlanet Karting - Playstation 3*. The 70B model gave a more cautious answer and explicitly stated that the reviews did not clearly mention the track quality. The 8B model was shorter and more direct.

For the query **“story-rich scary game with dark atmosphere”**, the 8B model recommended *The Dark Pictures: House of Ashes* directly. The 70B model suggested *Among the Sleep* and also mentioned *House of Ashes* as an alternative. The 70B answer was more detailed, but also less decisive.

For the query **“good football game with realistic teams”**, the 8B model took a conservative approach and said that the retrieved context did not provide enough information to confirm realistic teams. The 70B model tried to infer realism from indirect clues such as comparisons to Madden. That made it somewhat more helpful, but also slightly less strict in staying tied to the evidence.

For the query **“family-friendly game for casual players”**, both models produced useful results. The 8B model returned multiple suggestions, while the 70B model focused on one strong recommendation. The 70B response was cleaner, but the 8B response gave the user more options.

For the query **“game with strong story and puzzle elements”**, both models recommended *The Book of Unwritten Tales*. The 70B model added an alternative title and more explanation, while the 8B model stayed shorter and more focused.

* Key observations:

  * `llama-3.3-70b-versatile` generally produced more detailed and nuanced responses.
  * `llama-3.1-8b-instant` was more concise and often more conservative when the evidence was weak.
  * The larger model was not clearly better for every query in this product-review setting.
  * In some cases, the extra detail from the 70B model came with mild over-interpretation beyond the retrieved evidence.
  * For this project, groundedness, response speed, and practical API usage are all important, not just answer length.

We decided to keep `llama-3.1-8b-instant` as the default model in the pipeline. Although `llama-3.3-70b-versatile` often produced richer answers, the 8B model remained more concise, more conservative when the retrieved evidence was limited, and more practical for a review-grounded shopping assistant. For this application, those qualities were more valuable than longer or more elaborate responses.


## Step 2: Additional Feature (state which option you chose)

### What You Implemented

- Description of the feature
- Key results or examples
  
## Step 3: Improve Documentation and Code Quality

- created src/config.py and updated build_retriver.py to remove hardcoded path
- remove api key from .env.example
- ensure all functions have docstring


### Documentation Update
- Summary of `README` improvements

### Code Quality Changes
- Summary of cleanups

## Step 4: Cloud Deployment Plan
(See Step 4 above for required subsections)