package rag

import (
	"context"
	"hash/fnv"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/0xSalik/recall/internal/chunker"
	"github.com/0xSalik/recall/internal/embed"
	"github.com/0xSalik/recall/internal/store"
)

// fakeEmbedder is a deterministic bag-of-words embedder (shared-token overlap
// drives similarity), good enough to validate retrieval ordering.
type fakeEmbedder struct{ dims int }

func (f fakeEmbedder) Dims() int { return f.dims }

func (f fakeEmbedder) Embed(texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i, t := range texts {
		v := make([]float32, f.dims)
		for _, tok := range strings.Fields(strings.ToLower(t)) {
			h := fnv.New32a()
			h.Write([]byte(tok))
			v[h.Sum32()%uint32(f.dims)] += 1
		}
		// L2 normalize.
		var sum float64
		for _, x := range v {
			sum += float64(x) * float64(x)
		}
		if sum > 0 {
			inv := float32(1.0 / sqrt(sum))
			for j := range v {
				v[j] *= inv
			}
		}
		out[i] = v
	}
	return out, nil
}

func sqrt(x float64) float64 {
	// avoid importing math just for tests of this size
	z := x
	for i := 0; i < 40; i++ {
		z = (z + x/z) / 2
	}
	return z
}

// fakeGenerator echoes the question and a marker so tests can assert the prompt
// reached generation with context.
type fakeGenerator struct{ lastPrompt string }

func (g *fakeGenerator) Generate(prompt string, opts GenOptions) (string, error) {
	g.lastPrompt = prompt
	return "ANSWER based on context", nil
}

func (g *fakeGenerator) GenerateStream(ctx context.Context, prompt string, opts GenOptions, tokens chan<- string) (string, error) {
	g.lastPrompt = prompt
	for _, w := range []string{"ANSWER", " based", " on", " context"} {
		tokens <- w
	}
	return "ANSWER based on context", nil
}

var _ embed.Embedder = fakeEmbedder{}

func newTestRAG(t *testing.T) (*RAG, *fakeGenerator) {
	t.Helper()
	st, err := store.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	gen := &fakeGenerator{}
	r := NewWithComponents(st, fakeEmbedder{dims: 256}, gen)
	return r, gen
}

func TestIndexAndQuery(t *testing.T) {
	dir := t.TempDir()
	mustWrite(t, dir, "hnsw.md", "# HNSW\n\nHNSW construction assigns each node a random layer and connects to nearest neighbors.")
	mustWrite(t, dir, "cooking.md", "# Cooking\n\nTo bake bread you need flour water yeast and salt mixed together.")

	r, gen := newTestRAG(t)
	results, err := r.Index([]string{dir})
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 2 {
		t.Fatalf("expected 2 indexed files, got %d", len(results))
	}
	if r.Store().ChunkCount() == 0 {
		t.Fatal("no chunks indexed")
	}

	ans, err := r.Ask("how does hnsw construction assign layers to nodes", 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(ans.Sources) == 0 {
		t.Fatal("no sources returned")
	}
	// The HNSW note should rank above the cooking note.
	top := ans.Sources[0].Chunk
	if !strings.Contains(strings.ToLower(top.Text), "hnsw") {
		t.Fatalf("expected HNSW chunk on top, got: %q", top.Text)
	}
	// The prompt handed to generation must contain the question and context.
	if !strings.Contains(gen.lastPrompt, "Question:") || !strings.Contains(gen.lastPrompt, "Context:") {
		t.Fatal("prompt missing expected structure")
	}
	if !strings.Contains(gen.lastPrompt, "[Source:") {
		t.Fatal("prompt missing source attribution")
	}
}

func TestIndexIdempotent(t *testing.T) {
	dir := t.TempDir()
	mustWrite(t, dir, "a.txt", strings.Repeat("alpha beta gamma delta. ", 30))

	r, _ := newTestRAG(t)
	if _, err := r.Index([]string{dir}); err != nil {
		t.Fatal(err)
	}
	first := r.Store().ChunkCount()

	// Re-index the same unchanged directory: chunk count must not grow.
	res, err := r.Index([]string{dir})
	if err != nil {
		t.Fatal(err)
	}
	if r.Store().ChunkCount() != first {
		t.Fatalf("re-index changed chunk count: %d -> %d", first, r.Store().ChunkCount())
	}
	if len(res) != 1 || !res[0].Skipped {
		t.Fatalf("expected the file to be skipped on re-index, got %+v", res)
	}
}

func TestReindexChangedFileNoDuplicates(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "a.txt")
	mustWriteFile(t, p, strings.Repeat("alpha beta gamma. ", 30))

	r, _ := newTestRAG(t)
	if _, err := r.Index([]string{dir}); err != nil {
		t.Fatal(err)
	}
	first := r.Store().ChunkCount()
	if first == 0 {
		t.Fatal("nothing indexed")
	}

	// Modify the file (and bump mtime) then re-index: chunk count should
	// reflect the new content, not stack on top of the old chunks.
	mustWriteFile(t, p, strings.Repeat("delta epsilon zeta eta. ", 30))
	bumpModTime(t, p)
	if _, err := r.Index([]string{dir}); err != nil {
		t.Fatal(err)
	}
	if r.Store().FileCount() != 1 {
		t.Fatalf("file count = %d, want 1", r.Store().FileCount())
	}
	// All surviving chunks must be from the new content.
	for _, f := range r.Store().ListFiles() {
		if f.Path == p && f.Chunks == 0 {
			t.Fatal("file has no chunks after reindex")
		}
	}
	ans, err := r.Ask("epsilon zeta", 5)
	if err != nil {
		t.Fatal(err)
	}
	for _, s := range ans.Sources {
		if strings.Contains(s.Chunk.Text, "alpha beta gamma") {
			t.Fatal("stale chunk from old file content still present after reindex")
		}
	}
}

func TestRefreshPrunesDeletedFiles(t *testing.T) {
	dir := t.TempDir()
	keep := filepath.Join(dir, "keep.txt")
	gone := filepath.Join(dir, "gone.txt")
	mustWriteFile(t, keep, "vectors are embedded and indexed for retrieval ranking")
	mustWriteFile(t, gone, "sourdough needs flour water salt and a starter culture")

	r, _ := newTestRAG(t)
	if _, err := r.Index([]string{dir}); err != nil {
		t.Fatal(err)
	}
	if r.Store().FileCount() != 2 {
		t.Fatalf("expected 2 files, got %d", r.Store().FileCount())
	}

	// Delete one file on disk, then refresh.
	if err := os.Remove(gone); err != nil {
		t.Fatal(err)
	}
	res, err := r.Refresh(nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Deleted) != 1 || res.Deleted[0] != gone {
		t.Fatalf("expected gone.txt to be pruned, got %+v", res.Deleted)
	}
	if r.Store().FileCount() != 1 {
		t.Fatalf("file count after refresh = %d, want 1", r.Store().FileCount())
	}
	// A query must no longer surface the deleted file.
	ans, err := r.Ask("how do I bake sourdough?", 5)
	if err != nil {
		t.Fatal(err)
	}
	for _, s := range ans.Sources {
		if s.Chunk.Source == gone {
			t.Fatal("deleted file still referenced in query results")
		}
	}
}

func TestRemoveAndClear(t *testing.T) {
	dir := t.TempDir()
	sub := filepath.Join(dir, "sub")
	if err := os.MkdirAll(sub, 0o755); err != nil {
		t.Fatal(err)
	}
	mustWriteFile(t, filepath.Join(dir, "top.txt"), strings.Repeat("top level content about vectors and embeddings. ", 5))
	mustWriteFile(t, filepath.Join(sub, "deep.txt"), strings.Repeat("nested content about retrieval and ranking. ", 5))

	r, _ := newTestRAG(t)
	if _, err := r.Index([]string{dir}); err != nil {
		t.Fatal(err)
	}
	if r.Store().FileCount() != 2 {
		t.Fatalf("expected 2 files, got %d", r.Store().FileCount())
	}

	// Remove the sub/ directory.
	n, files, err := r.Remove(sub)
	if err != nil {
		t.Fatal(err)
	}
	if n == 0 || len(files) != 1 {
		t.Fatalf("remove returned n=%d files=%v", n, files)
	}
	if r.Store().FileCount() != 1 {
		t.Fatalf("file count after remove = %d, want 1", r.Store().FileCount())
	}

	// Clear wipes the rest.
	if err := r.Clear(); err != nil {
		t.Fatal(err)
	}
	if r.Store().ChunkCount() != 0 || r.Store().FileCount() != 0 {
		t.Fatalf("store not empty after clear: %d chunks, %d files", r.Store().ChunkCount(), r.Store().FileCount())
	}
}

func TestRetrieveOnly(t *testing.T) {
	dir := t.TempDir()
	mustWriteFile(t, filepath.Join(dir, "h.md"), "# HNSW\n\nHNSW is a graph based vector index for approximate nearest neighbor search")
	r, _ := newTestRAG(t)
	if _, err := r.Index([]string{dir}); err != nil {
		t.Fatal(err)
	}
	sources, err := r.Retrieve("graph vector index", 3)
	if err != nil {
		t.Fatal(err)
	}
	if len(sources) == 0 {
		t.Fatal("expected retrieval results")
	}
}

func TestAskStream(t *testing.T) {
	dir := t.TempDir()
	mustWrite(t, dir, "x.txt", "vectors are embedded then indexed for retrieval.")
	r, _ := newTestRAG(t)
	if _, err := r.Index([]string{dir}); err != nil {
		t.Fatal(err)
	}

	tokens := make(chan string, 16)
	var got strings.Builder
	done := make(chan struct{})
	go func() {
		for tok := range tokens {
			got.WriteString(tok)
		}
		close(done)
	}()
	ans, err := r.AskStream(context.Background(), "how are vectors retrieved", 1, tokens)
	close(tokens)
	<-done
	if err != nil {
		t.Fatal(err)
	}
	if got.String() != "ANSWER based on context" {
		t.Fatalf("streamed text mismatch: %q", got.String())
	}
	if ans.Text != "ANSWER based on context" {
		t.Fatalf("assembled text mismatch: %q", ans.Text)
	}
}

func TestBuildPromptStructure(t *testing.T) {
	src := []Source{
		{Chunk: chunk("notes.md", 0, "first chunk text"), Score: 0.9},
		{Chunk: chunk("thesis.pdf", 7, "second chunk text"), Score: 0.8},
	}
	p := buildPrompt("what is x?", src)
	if !strings.Contains(p, "[Source: notes.md]") {
		t.Fatal("missing non-paginated source line")
	}
	if !strings.Contains(p, "[Source: thesis.pdf, Page 7]") {
		t.Fatal("missing paginated source line")
	}
	if !strings.HasSuffix(p, "Answer:") {
		t.Fatal("prompt should end with Answer:")
	}
}

func TestCleanAnswer(t *testing.T) {
	if got := cleanAnswer("  hello world [end of text] garbage"); got != "hello world" {
		t.Fatalf("cleanAnswer = %q", got)
	}
	if got := cleanAnswer("answer<|im_end|>"); got != "answer" {
		t.Fatalf("cleanAnswer = %q", got)
	}
}

// helpers

func mustWrite(t *testing.T, dir, name, content string) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(dir, name), []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

func chunk(source string, page int, text string) chunker.Chunk {
	return chunker.Chunk{ID: source + text, Source: source, PageNum: page, Text: text}
}

func mustWriteFile(t *testing.T, path, content string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

func bumpModTime(t *testing.T, path string) {
	t.Helper()
	future := time.Now().Add(2 * time.Second)
	if err := os.Chtimes(path, future, future); err != nil {
		t.Fatal(err)
	}
}
