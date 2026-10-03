# Put your files here

| File | Required | What it is |
|---|---|---|
| `chunks.jsonl` | yes | One JSON object per line (see `chunks.example.jsonl`) |
| `embeddings.npy` | yes | NumPy float32 array, shape (number_of_chunks, dimension) |
| `embedding_info.json` | strongly recommended | Which model made the embeddings (see `embedding_info.example.json`) |
| `metadata.jsonl` | optional | Extra fields per chunk, matched by `chunk_id` |

## The rule that prevents every compatibility problem
Row `i` of `embeddings.npy` belongs to line `i` of `chunks.jsonl`.

## Fields in chunks.jsonl
- `chunk_id` (text, unique) **required**
- `text` (text) **required**
- `source` (paper name, e.g. the PDF file name) **required**
- `page` (integer) strongly recommended - used for citations
- `section` (e.g. "Results") optional - helps skip introductions/references during extraction
- `chunk_type` (`"text"` or `"table"`) optional, default `"text"`

Delete the two `*.example.*` files or leave them; the app ignores them.
