// Package rag wires the pieces together into a retrieval-augmented generation
// pipeline: ingest -> chunk -> embed -> index on the write path, and
// embed -> retrieve -> generate on the read path.
package rag

import (
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/0xSalik/recall/internal/chunker"
	"github.com/0xSalik/recall/internal/embed"
	"github.com/0xSalik/recall/internal/ingest"
	"github.com/0xSalik/recall/internal/store"
)

// embedBatch bounds how many chunks we embed per backend call during ingest.
const embedBatch = 32

// tokenWarnThreshold is the approximate prompt-token count above which we warn
// about possibly overflowing a small model's context window.
const tokenWarnThreshold = 3000

// Source is a retrieved chunk paired with its similarity score.
type Source struct {
	Chunk chunker.Chunk
	Score float32
}

// Answer is the result of a query: the generated text plus the chunks it was
// grounded in.
type Answer struct {
	Text    string
	Sources []Source
}

// RAG owns the store, the embedder, and the generator.
type RAG struct {
	store     *store.Store
	embedder  embed.Embedder
	generator Generator
	exts      []string
	chunkCfg  chunker.Config
}

// New opens (or creates) the store at storePath, detects an embedding backend
// for embedModel, and prepares the llama-cli generator for genModel.
func New(storePath, embedModel, genModel, llamaPath string) (*RAG, error) {
	st, err := store.Open(storePath)
	if err != nil {
		return nil, err
	}
	embedder, err := embed.Detect(embedModel, "llama-embedding", "http://localhost:8081")
	if err != nil {
		return nil, err
	}
	gen, err := NewLlamaGenerator(genModel, llamaPath)
	if err != nil {
		return nil, err
	}
	return &RAG{store: st, embedder: embedder, generator: gen}, nil
}

// NewIndexer opens the store and an embedding backend, but no generator. Use it
// for commands that write to or read the index without generating text (index,
// refresh, search), so they don't require the generation binary/model.
func NewIndexer(storePath, embedModel string) (*RAG, error) {
	st, err := store.Open(storePath)
	if err != nil {
		return nil, err
	}
	embedder, err := embed.Detect(embedModel, "llama-embedding", "http://localhost:8081")
	if err != nil {
		return nil, err
	}
	return &RAG{store: st, embedder: embedder}, nil
}

// NewManager opens only the store, for management commands (status, list, clear,
// remove) that touch neither embeddings nor generation.
func NewManager(storePath string) (*RAG, error) {
	st, err := store.Open(storePath)
	if err != nil {
		return nil, err
	}
	return &RAG{store: st}, nil
}

// NewWithComponents builds a RAG from already-constructed parts. It is the
// injection point for tests (fake embedder/generator) and for callers that want
// to share an embedder between indexing and querying.
func NewWithComponents(st *store.Store, embedder embed.Embedder, gen Generator) *RAG {
	return &RAG{store: st, embedder: embedder, generator: gen}
}

// SetExtensions restricts directory ingestion to the given extensions.
func (r *RAG) SetExtensions(exts []string) { r.exts = exts }

// Store exposes the underlying store (for status/serve).
func (r *RAG) Store() *store.Store { return r.store }

// IndexResult reports per-path ingestion outcome for CLI progress output.
type IndexResult struct {
	Path    string
	Chunks  int
	Pages   int
	Skipped bool
}

// Index ingests, chunks, embeds, and stores each path (file or directory). Files
// that are already indexed and unchanged are skipped; files that changed since
// they were indexed have their old chunks removed before re-indexing (so
// re-running index never duplicates chunks). The store is saved once at the end.
func (r *RAG) Index(paths []string) ([]IndexResult, error) {
	docs, results, err := r.collect(paths)
	if err != nil {
		return results, err
	}

	// Any doc we're about to index that's already in the manifest is a changed
	// file; drop its stale chunks in a single rebuild before re-adding.
	var changed []string
	for _, d := range docs {
		if r.store.IsIndexed(d.Path) {
			changed = append(changed, d.Path)
		}
	}
	if len(changed) > 0 {
		if _, rerr := r.store.RemoveFiles(changed); rerr != nil {
			return results, rerr
		}
	}

	for _, doc := range docs {
		res, ierr := r.indexDoc(doc)
		if ierr != nil {
			return results, ierr
		}
		results = append(results, res)
	}
	if err := r.store.Save(); err != nil {
		return results, err
	}
	return results, nil
}

// collect resolves paths into the documents that need (re)indexing, recording
// skipped (unchanged) files in the returned results. A file reachable via more
// than one input path (e.g. a directory plus an explicit file) is collected
// once.
func (r *RAG) collect(paths []string) ([]ingest.Document, []IndexResult, error) {
	var docs []ingest.Document
	var results []IndexResult
	seen := map[string]bool{}
	add := func(doc ingest.Document) {
		if seen[doc.Path] {
			return
		}
		seen[doc.Path] = true
		docs = append(docs, doc)
	}
	for _, p := range paths {
		info, err := os.Stat(p)
		if err != nil {
			return nil, results, fmt.Errorf("rag: %s: %w", p, err)
		}
		if info.IsDir() {
			dirDocs, derr := ingest.IngestDir(p, r.exts)
			if derr != nil {
				return nil, results, derr
			}
			for _, doc := range dirDocs {
				if r.store.HasFile(doc.Path, doc.Modified) {
					results = append(results, IndexResult{Path: doc.Path, Skipped: true})
					continue
				}
				add(doc)
			}
		} else {
			if r.store.HasFile(p, info.ModTime()) {
				results = append(results, IndexResult{Path: p, Skipped: true})
				continue
			}
			doc, ierr := ingest.IngestFile(p)
			if ierr != nil {
				return nil, results, ierr
			}
			add(doc)
		}
	}
	return docs, results, nil
}

// ListFiles returns the indexed files with chunk counts.
func (r *RAG) ListFiles() []store.FileInfo { return r.store.ListFiles() }

// Clear empties the store and persists the change.
func (r *RAG) Clear() error {
	r.store.Clear()
	return r.store.Save()
}

// Remove deletes everything indexed at target (an exact file or, treating it as
// a directory, everything beneath it) and persists. It returns the number of
// chunks removed and the files removed.
func (r *RAG) Remove(target string) (int, []string, error) {
	n, files, err := r.store.Remove(target)
	if err != nil {
		return n, files, err
	}
	return n, files, r.store.Save()
}

// RefreshResult summarizes a refresh: files dropped because they no longer exist
// on disk, and the per-file results of re-indexing changed/new files.
type RefreshResult struct {
	Deleted   []string
	Reindexed []IndexResult
}

// Refresh re-syncs the store with the filesystem. It always prunes chunks for
// indexed files that no longer exist on disk and re-indexes files that changed.
// If paths are given, it also discovers and indexes new files under them.
func (r *RAG) Refresh(paths []string) (RefreshResult, error) {
	var deleted, changed []string
	for _, f := range r.store.ListFiles() {
		info, err := os.Stat(f.Path)
		if err != nil {
			if os.IsNotExist(err) {
				deleted = append(deleted, f.Path)
			}
			continue
		}
		if !r.store.HasFile(f.Path, info.ModTime()) {
			changed = append(changed, f.Path)
		}
	}
	if len(deleted) > 0 {
		if _, err := r.store.RemoveFiles(deleted); err != nil {
			return RefreshResult{}, err
		}
	}
	// Index dedups changed files (removes stale chunks first) and saves the
	// store, which also persists the deletions above even when nothing is added.
	reindex := append(append([]string{}, paths...), changed...)
	results, err := r.Index(reindex)
	return RefreshResult{Deleted: deleted, Reindexed: results}, err
}

// Retrieve embeds the question and returns the top-k matching chunks without
// running generation. Useful for fast, model-free inspection of what the index
// would feed the LLM.
func (r *RAG) Retrieve(question string, k int) ([]Source, error) {
	return r.retrieve(question, k)
}

// indexDoc chunks, embeds, and stores a single document.
func (r *RAG) indexDoc(doc ingest.Document) (IndexResult, error) {
	if r.store.HasFile(doc.Path, doc.Modified) {
		return IndexResult{Path: doc.Path, Skipped: true}, nil
	}

	var chunks []chunker.Chunk
	if doc.Format == "pdf" && len(doc.Pages) > 0 {
		// Chunk each page separately so PageNum can be attributed.
		for i, page := range doc.Pages {
			if strings.TrimSpace(page) == "" {
				continue
			}
			pageChunks := chunker.Split(page, doc.Path, r.chunkCfg)
			for j := range pageChunks {
				pageChunks[j].PageNum = i + 1
			}
			chunks = append(chunks, pageChunks...)
		}
	} else {
		chunks = chunker.Split(doc.Content, doc.Path, r.chunkCfg)
	}

	if len(chunks) == 0 {
		// Nothing extractable; still mark indexed so we don't retry every run.
		r.store.MarkIndexed(doc.Path, doc.Modified)
		return IndexResult{Path: doc.Path, Chunks: 0, Pages: len(doc.Pages)}, nil
	}

	// Embed in batches and add to the store.
	texts := make([]string, len(chunks))
	for i, c := range chunks {
		texts[i] = c.Text
	}
	vecs := make([][]float32, 0, len(texts))
	for i := 0; i < len(texts); i += embedBatch {
		end := i + embedBatch
		if end > len(texts) {
			end = len(texts)
		}
		batch, err := r.embedder.Embed(texts[i:end])
		if err != nil {
			return IndexResult{}, fmt.Errorf("rag: embedding %s: %w", doc.Path, err)
		}
		vecs = append(vecs, batch...)
	}
	if err := r.store.AddChunks(chunks, vecs); err != nil {
		return IndexResult{}, err
	}
	r.store.MarkIndexed(doc.Path, doc.Modified)
	return IndexResult{Path: doc.Path, Chunks: len(chunks), Pages: len(doc.Pages)}, nil
}

// retrieve embeds the question and returns the top-k chunks as Sources.
func (r *RAG) retrieve(question string, k int) ([]Source, error) {
	vecs, err := r.embedder.Embed([]string{question})
	if err != nil {
		return nil, fmt.Errorf("rag: embedding question: %w", err)
	}
	if len(vecs) == 0 {
		return nil, fmt.Errorf("rag: question produced no embedding")
	}
	chunks, scores, err := r.store.Search(vecs[0], k)
	if err != nil {
		return nil, err
	}
	sources := make([]Source, len(chunks))
	for i := range chunks {
		sources[i] = Source{Chunk: chunks[i], Score: scores[i]}
	}
	return sources, nil
}

// Ask runs the full pipeline and returns the answer with its sources.
func (r *RAG) Ask(question string, k int) (Answer, error) {
	sources, err := r.retrieve(question, k)
	if err != nil {
		return Answer{}, err
	}
	prompt := buildPrompt(question, sources)
	text, err := r.generator.Generate(prompt, DefaultGenOptions())
	if err != nil {
		return Answer{}, err
	}
	return Answer{Text: text, Sources: sources}, nil
}

// AskStream runs retrieval, then streams generated tokens onto the tokens
// channel. It returns the assembled answer and sources. The caller owns and
// should range/close the channel appropriately.
func (r *RAG) AskStream(ctx context.Context, question string, k int, tokens chan<- string) (Answer, error) {
	sources, err := r.retrieve(question, k)
	if err != nil {
		return Answer{}, err
	}
	prompt := buildPrompt(question, sources)
	text, err := r.generator.GenerateStream(ctx, prompt, DefaultGenOptions(), tokens)
	if err != nil {
		return Answer{Text: text, Sources: sources}, err
	}
	return Answer{Text: text, Sources: sources}, nil
}

// Query satisfies the original design signature: it returns just the answer
// text. The stream flag is accepted for API compatibility; streaming consumers
// should use AskStream which exposes the token channel.
func (r *RAG) Query(question string, k int, stream bool) (string, error) {
	ans, err := r.Ask(question, k)
	if err != nil {
		return "", err
	}
	return ans.Text, nil
}

// buildPrompt assembles the grounded-QA prompt from retrieved sources. It warns
// to stderr if the prompt is large enough to risk overflowing a small context
// window.
func buildPrompt(question string, sources []Source) string {
	var b strings.Builder
	b.WriteString("You are a helpful assistant answering questions about the user's personal files.\n")
	b.WriteString("Answer based only on the provided context. If the context doesn't contain the answer, say so.\n\n")
	b.WriteString("Context:\n---\n")
	for _, s := range sources {
		if s.Chunk.PageNum > 0 {
			fmt.Fprintf(&b, "[Source: %s, Page %d]\n", s.Chunk.Source, s.Chunk.PageNum)
		} else {
			fmt.Fprintf(&b, "[Source: %s]\n", s.Chunk.Source)
		}
		b.WriteString(s.Chunk.Text)
		b.WriteString("\n\n")
	}
	b.WriteString("---\n\n")
	fmt.Fprintf(&b, "Question: %s\n\nAnswer:", question)

	prompt := b.String()
	// Rough token estimate: ~4 chars per token.
	if approxTokens := len(prompt) / 4; approxTokens > tokenWarnThreshold {
		fmt.Fprintf(os.Stderr, "warning: prompt is ~%d tokens, which may exceed a small model's context window\n", approxTokens)
	}
	return prompt
}
