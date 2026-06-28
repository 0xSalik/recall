# recall

`recall` is a local-first retrieval-augmented generation tool: it indexes your files into a vector store and answers questions over them using local llama.cpp models. Nothing leaves your machine — no OpenAI, no Pinecone, no LangChain. It is a single Go binary plus two GGUF model files.

## How it works

The pipeline has a write path and a read path, and they share a persistent store.

On the **write path** (`recall index`), each file is ingested into plain text. PDFs go through `ledongthuc/pdf` page by page (scanned, image-only pages are logged and skipped rather than silently producing empty content); markdown is run through a small state-machine stripper that removes syntax but keeps heading text and fenced code blocks verbatim, because code is semantically valuable and headings are useful context; source files are laid out so that function and class boundaries fall on blank lines, and each file is prefixed with a `// File: <path>` header so retrieved snippets are self-identifying. The extracted text is then **chunked**. Chunking respects natural boundaries: it splits on paragraphs first, falls back to sentence boundaries for oversized paragraphs, and only hard-splits at the character level as a last resort. Small adjacent paragraphs are merged up toward the target size, and every chunk carries a configurable overlap from the previous chunk so meaning that straddles a boundary isn't lost. Chunk IDs are a deterministic hash of content and source, which is what lets re-indexing be idempotent.

Each chunk is **embedded** into a unit-length vector by shelling out to llama.cpp (the `llama-embedding` CLI, or a `llama-server --embedding` HTTP endpoint as a fallback). Vectors are L2-normalized so cosine similarity reduces to a dot product. Vectors and chunks are written to a **store** under `~/.recall`: `chunks.json` holds the chunk structs, `index.bin` holds the serialized vector index, and `manifest.json` records `{path: modtime}` for change detection so unchanged files are skipped on the next run.

On the **read path** (`recall query`), the question is embedded with the same model, the **index** returns the top-k nearest chunks, and those chunks are assembled into a grounded-QA prompt that instructs the model to answer only from the provided context and to say so when the context doesn't contain the answer. Generation runs through the `llama-cli` subprocess; results can be returned whole or streamed token by token.

The **index** itself comes in two flavors behind one interface. `FlatIndex` is an exact brute-force cosine scan — O(n) per query, always correct, and the ground-truth baseline used to measure the approximate index. `HNSW` is a hierarchical navigable small-world graph built from scratch (Malkov & Yashunin, 2018): a layered graph where upper layers are sparse express lanes and layer 0 is dense. Construction assigns each node a random level, greedily descends to that level, then does an `ef`-bounded beam search at each remaining layer to pick neighbors via the selection *heuristic* (keep an edge only if the candidate is closer to the new node than to any already-selected neighbor — this keeps the graph navigable and is the difference between ~78% and ~94% recall@5). Layer 0 is intentionally denser (`Mmax0 = 2*M`). The graph stores neighbors as integer indices, not pointers, so it gob-serializes without custom marshaling.

## Install (prebuilt, zero setup)

Download the binary for your platform from the [Releases](https://github.com/0xSalik/recall/releases) page and run it — the llama.cpp engine and the embedding model are baked into the binary, and the generation model is downloaded on first `query`/`serve` (with a progress bar) into `~/.recall`. No PATH, no toolchain, no manual model downloads.

```bash
# macOS (one-time: clear the download quarantine on unsigned binaries)
xattr -d com.apple.quarantine ./recall-darwin-arm64 2>/dev/null || true
chmod +x ./recall-darwin-arm64
./recall-darwin-arm64 serve
```

Resolution order for the engine and models is: explicit flag (`--bin`/`--embed`/`--gen`) → cache in `~/.recall` → copy embedded in the binary → download. So a prebuilt binary "just works", while a source build (below) transparently falls back to downloading what it needs on first use.

## Setup (from source)

Prerequisites:

- Go 1.21+
- A C/C++ toolchain to build llama.cpp (or prebuilt llama.cpp binaries on your `PATH`)

A plain `make build` produces a small binary that downloads the engine model(s) on first use. To produce a fully self-contained, one-click binary like the release assets, run `make bundle` (needs `cmake` + a C/C++ toolchain): it builds CPU-only llama.cpp, embeds the engine binaries and the embedding model via `//go:embed` behind the `bundle` build tag, and leaves the large generation model to be fetched on first use (keeping the artifact under GitHub's 2 GB asset limit). CI does this per-platform in `.github/workflows/release.yml`.

Build llama.cpp (skip if you already have the binaries):

```bash
git clone https://github.com/ggerganov/llama.cpp third_party/llama.cpp
cd third_party/llama.cpp && cmake -B build && cmake --build build -j && cd ../..
export PATH="$PWD/third_party/llama.cpp/build/bin:$PATH"   # or pass --llama <path>
```

`recall` needs three binaries from that build on your `PATH`:

- `llama-embedding` — used for embeddings
- `llama-completion` — used for generation (current llama.cpp; on older builds the
  single `llama-cli` binary is used as a fallback)
- `llama-server` — optional, only for the embedding HTTP fallback

`third_party/` is git-ignored. If you'd rather not touch `PATH`, point `recall`
straight at the binary directory with `--bin` on any command:

```bash
./bin/recall index ~/Documents --bin third_party/llama.cpp/build/bin
```

`--bin` is prepended to `PATH` for both the embedding and generation binaries, so
it's the simplest way to run without editing your shell environment.

Download the models (optional — recall fetches them automatically on first use into `~/.recall/models`; use this only to pre-seed or to keep them in the repo dir):

```bash
make models
# embedding: nomic-embed-text-v1.5 Q4_K_M (~84MB)
# generation: Phi-3-mini-4k-instruct Q4_K_M (~2.3GB)
```

Build recall:

```bash
make build      # produces bin/recall (the browser UI is embedded via go:embed)
make test       # run the test suite
make bench      # Go benchmarks + standalone search/recall benchmark
```

## Usage

Index files or directories. Re-running only processes changed files:

```bash
$ recall index ~/Documents ~/code/myproject --ext .md,.pdf,.go,.py
Indexing...
  [1/47] notes.md (3 chunks)
  [2/47] thesis.pdf (142 chunks, 8 pages)
  ...
  Skipped 12 files (unchanged)
Done. 847 chunks indexed in 23.4s
```

Ask a question, optionally streaming and showing sources:

```bash
$ recall query "what did I write about HNSW construction in my notes?" --sources --stream
Answer: You described HNSW construction as assigning each node a random layer,
then connecting it to its nearest neighbors using a selection heuristic that
keeps the graph navigable...

Sources:
  notes/hnsw.md (score: 0.94)
  papers/malkov-hnsw.pdf, page 7 (score: 0.91)
```

Retrieval only — embed the query and print the matching chunks without invoking the generation model (fast, and doesn't need a gen model installed):

```bash
$ recall search "HNSW construction" -k 3
1. [0.941] notes/hnsw.md
   HNSW assigns each node a random layer, then connects it to its nearest…
```

Manage the index. The index is incremental, so it also needs pruning and refreshing:

```bash
$ recall list                       # every indexed file with its chunk count
$ recall refresh                    # drop files deleted on disk, reindex changed ones
$ recall refresh ~/Documents        # ...and also pick up new files under a path
$ recall remove ~/code/old-project  # drop a single file or a whole folder
$ recall clear                      # wipe the index (prompts; --yes to skip)
```

`refresh` exists because re-running `index` only ever *adds* — it can't know a file was deleted or moved. `refresh` reconciles the store with the filesystem: indexed files that no longer exist are pruned, changed files are re-embedded (their stale chunks removed first, so nothing is duplicated), and any new files under the given paths are added. `index` itself now also removes a file's old chunks before re-indexing it when its mtime changed, so re-indexing never leaves duplicate chunks behind.

Check the store, or serve the browser UI + API:

```bash
$ recall status
Store: /Users/me/.recall
Files indexed: 47
Chunks: 847
Index type: HNSW
Store size: 12.3 MB

$ recall serve
recall serving on http://localhost:8080
```

The server exposes `GET /status`, `GET /files`, `POST /remove`, `POST /refresh`, `POST /clear`, and `POST /query` which content-negotiates: send `Accept: text/event-stream` for token-by-token SSE (what the browser UI uses), or omit it for a single JSON `{answer, sources}` response (what the CLI uses). The browser UI has a **manage index** panel (top right) to list, remove, refresh, and clear without leaving the page.

Deletion note: neither the flat nor the HNSW index supports in-place removal, so `remove`/`refresh`/`clear` work by rebuilding the index from the surviving vectors (extracted from the current index via `Entries()`). For the target corpus sizes this is cheap and keeps the index types simple.

## Architecture decisions and tradeoffs

**Subprocess over CGO for llama.cpp.** Binding llama.cpp through CGO means a C++ toolchain in the build, fragile linking, and platform-specific pain. Shelling out to `llama-cli` / `llama-embedding` keeps `recall` a pure-Go build that compiles anywhere, decouples it from llama.cpp's churning API, and makes the model process easy to reason about. The cost is process startup latency per call, which is negligible next to embedding and generation time.

**HNSW from scratch over FAISS.** FAISS would mean a C++ dependency and the build problems that come with it. A hand-written HNSW is a few hundred lines of Go, builds everywhere, and — with the selection heuristic and a tuned `ef` — gets ~94% recall@5 at 10k vectors while being roughly 8x faster than the exact scan. For this use case that recall is fine; you are reading the top chunks into a prompt, not doing exact kNN classification.

**The flat index stays.** It's the correctness baseline the HNSW recall numbers are measured against, and for corpora under ~10k chunks the brute-force scan is sub-2ms — there's no reason to reach for the approximate index, so both are available.

**Chunking strategy.** Embedding models capture meaning best over coherent passages, so the chunker works hard to cut on paragraph and sentence boundaries instead of fixed windows, and carries overlap so a fact split across a boundary still appears whole in at least one chunk. Deterministic IDs make the manifest-based dedup reliable across re-indexing.

**SSE over WASM for the UI.** An earlier design compiled the UI to WebAssembly. That was the wrong tool: the WASM binary can't do file I/O or run subprocesses, so it can't run inference anyway — it would just be a fetch client shipped as a megabyte of `.wasm`. Inference runs natively in the `serve` binary; the UI is a single embedded HTML file with vanilla JS that reads the response stream with `fetch` + `getReader()`. Streaming is where the real UX value is — tokens appear as the model produces them — and there's zero binary overhead or build step in the browser.

## What's missing vs production

- **Re-ranking.** A cross-encoder pass over the top-k retrieved chunks would improve answer quality beyond what bi-encoder cosine retrieval gives.
- **Incremental HNSW updates.** Deletes and large mutations currently want a rebuild; production HNSW needs tombstoning and incremental repair.
- **Multi-user / access control.** This is single-user and local by design; there's no auth, tenancy, or per-document permissions.
- **GPU inference.** Everything runs on CPU through llama.cpp; wiring up Metal/CUDA offload would cut generation latency substantially.

## Benchmark results

Search-layer numbers from `make bench` on an Apple M4 Pro, 128-dim vectors, clustered to mimic real embedding distributions (500 queries):

```
Index search latency (k=5), clustered vectors:
  n        flat (p50/p95/p99)               hnsw (p50/p95/p99)               recall@5
  1000     131µs / 158µs / 254µs            107µs / 136µs / 230µs            100.0%
  10000    1.60ms / 1.87ms / 2.04ms         185µs / 268µs / 430µs            94.0%
```

At 10k vectors HNSW is ~8x faster than the exact scan while keeping 94% recall@5; at 1k the corpus is small enough that the two are comparable and recall is exact. Embedding throughput and end-to-end generation latency depend on the chosen GGUF models and your CPU; run `recall query --stream` with a model installed to observe first-token and total latency live.
