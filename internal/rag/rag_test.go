package rag

import (
	"context"
	"hash/fnv"
	"os"
	"path/filepath"
	"strings"
	"testing"

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
