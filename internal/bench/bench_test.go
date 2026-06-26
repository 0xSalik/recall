// Package bench holds standalone benchmarks for the search and pipeline hot
// paths. Embedding/generation benchmarks use lightweight fakes so the suite
// runs without models present; the real-model numbers are produced by the
// standalone cmd/bench tool.
package bench

import (
	"context"
	"hash/fnv"
	"math"
	"math/rand"
	"strconv"
	"strings"
	"testing"

	"github.com/0xSalik/recall/internal/chunker"
	"github.com/0xSalik/recall/internal/index"
	"github.com/0xSalik/recall/internal/rag"
	"github.com/0xSalik/recall/internal/store"
)

func makeChunks(texts []string) []chunker.Chunk {
	chunks := make([]chunker.Chunk, len(texts))
	for i, t := range texts {
		chunks[i] = chunker.Chunk{
			ID:     "chunk-" + strconv.Itoa(i),
			Source: "/synthetic/doc.txt",
			Text:   t,
		}
	}
	return chunks
}

func unitVec(r *rand.Rand, dims int) []float32 {
	v := make([]float32, dims)
	var sum float64
	for i := range v {
		v[i] = float32(r.NormFloat64())
		sum += float64(v[i]) * float64(v[i])
	}
	inv := float32(1 / math.Sqrt(sum))
	for i := range v {
		v[i] *= inv
	}
	return v
}

func buildIndexes(n, dims int) (*index.FlatIndex, *index.HNSW, [][]float32) {
	r := rand.New(rand.NewSource(1))
	flat := index.NewFlatIndex(dims)
	h := index.NewHNSW(dims)
	vecs := make([][]float32, n)
	for i := 0; i < n; i++ {
		v := unitVec(r, dims)
		vecs[i] = v
		id := "id-" + strconv.Itoa(i)
		flat.Add(id, v, i)
		h.Add(id, v, i)
	}
	return flat, h, vecs
}

func BenchmarkFlatSearch(b *testing.B) {
	flat, _, _ := buildIndexes(10000, 128)
	r := rand.New(rand.NewSource(2))
	q := unitVec(r, 128)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		flat.Search(q, 5)
	}
}

func BenchmarkHNSWSearch(b *testing.B) {
	_, h, _ := buildIndexes(10000, 128)
	r := rand.New(rand.NewSource(2))
	q := unitVec(r, 128)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		h.Search(q, 5)
	}
}

// fakeEmbedder is a deterministic bag-of-words embedder for benchmarking the
// harness around embedding without a model.
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
		var sum float64
		for _, x := range v {
			sum += float64(x) * float64(x)
		}
		if sum > 0 {
			inv := float32(1 / math.Sqrt(sum))
			for j := range v {
				v[j] *= inv
			}
		}
		out[i] = v
	}
	return out, nil
}

func BenchmarkEmbedBatch32(b *testing.B) {
	e := fakeEmbedder{dims: 768}
	batch := make([]string, 32)
	for i := range batch {
		batch[i] = "the quick brown fox jumps over the lazy dog number " + strconv.Itoa(i)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		e.Embed(batch)
	}
}

type fakeGenerator struct{}

func (fakeGenerator) Generate(prompt string, opts rag.GenOptions) (string, error) {
	return "benchmark answer", nil
}
func (fakeGenerator) GenerateStream(ctx context.Context, prompt string, opts rag.GenOptions, tokens chan<- string) (string, error) {
	tokens <- "benchmark answer"
	return "benchmark answer", nil
}

func BenchmarkFullQuery(b *testing.B) {
	st, err := store.Open(b.TempDir())
	if err != nil {
		b.Fatal(err)
	}
	e := fakeEmbedder{dims: 256}
	// Seed the store with synthetic chunks.
	r := rag.NewWithComponents(st, e, fakeGenerator{})
	texts := make([]string, 500)
	for i := range texts {
		texts[i] = "document chunk about topic " + strconv.Itoa(i%50) + " with some filler words"
	}
	vecs, _ := e.Embed(texts)
	chunks := makeChunks(texts)
	if err := st.AddChunks(chunks, vecs); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := r.Ask("tell me about topic 7", 5); err != nil {
			b.Fatal(err)
		}
	}
}
