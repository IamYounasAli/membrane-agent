# Paper Research Agent (free, Streamlit)

Ask questions about your own scientific papers (library mode) or search papers and the web (global mode).
Everything runs on free tiers. API keys live only in **Streamlit secrets**.

## 1. Add your files (exact names)

Put these in `data/imported/` :

| File | What it is |
|---|---|
| `chunks.jsonl` | one JSON object per line: `chunk_id`, `text`, `source`, `page`, optional `section`, `chunk_type` |
| `embeddings.npy` | float32 array, shape `(number_of_chunks, dimension)` |
| `embedding_info.json` | `{"model": "BAAI/bge-small-en-v1.5", "dimension": 384, "normalized": true, "query_prefix": ""}` |

**The one rule:** row *i* of `embeddings.npy` belongs to line *i* of `chunks.jsonl`.

**Use the same model for questions as you used for the chunks.** Put your model's exact name in
`embedding_info.json`. Supported out of the box: any model that `fastembed` provides
(e.g. `BAAI/bge-small-en-v1.5`, `BAAI/bge-base-en-v1.5`, `sentence-transformers/all-MiniLM-L6-v2`).
If you used another model (e.g. `all-mpnet-base-v2`), add `sentence-transformers` to `requirements.txt` and set
`EMBEDDING_BACKEND = "sentence_transformers"` in Streamlit secrets.

Convert what you already have (run once in Colab or any Python):

```python
import json, numpy as np
# chunks: list of dicts (chunk_id, text, source, page, ...);  vectors: same order
with open("chunks.jsonl", "w", encoding="utf-8") as f:
    for c in chunks:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")
np.save("embeddings.npy", np.asarray(vectors, dtype="float32"))
json.dump({"model": "BAAI/bge-small-en-v1.5", "dimension": 384, "normalized": True},
          open("embedding_info.json", "w"))
```

Size: keep each file under 100 MB (plain GitHub, no Git LFS). 5,000 chunks at 384 dimensions is about 8 MB.

## 2. Deploy

1. Push this folder to a GitHub repository (the `.gitignore` already keeps secrets out).
2. On share.streamlit.io choose the repo, main file `app.py`.
3. App → **Settings → Secrets**, paste (see `.streamlit/secrets.toml.example`):

```toml
GEMINI_API_KEY = "..."       # free: aistudio.google.com/apikey
GROQ_API_KEY = "..."         # optional backup (free)
OPENALEX_API_KEY = "..."     # free account at openalex.org
TAVILY_API_KEY = "..."       # free 1,000 credits/month
```

4. Open the app → **Data check** page → press the test buttons.

## 3. Use it

1. **Extract records** (once per paper): builds the structured database of measured values.
2. **Ask**: pick *Search my library* or *Search globally*; the page suggests what to type.
3. Streamlit Cloud forgets files on restart: on the Extract page download `records_seed.json` and commit it to `data/records_seed.json`.

## How it saves tokens
Embedding runs locally (no LLM). Only result/table/number-heavy chunks are extracted, once per paper.
A question costs at most 2 LLM calls. Answers come from the structured database first, then a few excerpts.

## Free-tier notes
Limits change; check them in each provider's console. If one LLM hits its limit the app falls back to the next
configured one. Free Gemini may use prompts to improve Google products, so keep confidential papers out or
run a local model.

## Change the topic
Everything topic-specific (fields, prompts, suggested questions) is in `core/schema.py`.

## Known limits
- "Add to library" from global results is not built in: download the open-access PDF and run it through your own
  chunking/embedding, then append to the files in `data/imported/`.
- Values inside figures cannot be extracted; tables only if your chunks kept them readable.
