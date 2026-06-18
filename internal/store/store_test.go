package store

import (
	"math"
	"math/rand"
	"testing"
	"time"

	"github.com/0xSalik/recall/internal/chunker"
)

func unitVec(r *rand.Rand, dims int) []float32 {
	v := make([]float32, dims)
	var sum float64
	for i := range v {
		v[i] = float32(r.NormFloat64())
		sum += float64(v[i]) * float64(v[i])
	}
	inv := float32(1.0 / math.Sqrt(sum))
	for i := range v {
		v[i] *= inv
	}
	return v
}

func makeChunks(n int) ([]chunker.Chunk, [][]float32) {
	r := rand.New(rand.NewSource(1))
	chunks := make([]chunker.Chunk, n)
	vecs := make([][]float32, n)
	for i := range chunks {
		chunks[i] = chunker.Chunk{
			ID:     "chunk-" + itoa(i),
			Source: "/docs/file.txt",
			Text:   "chunk text number " + itoa(i),
		}
		vecs[i] = unitVec(r, 48)
	}
	return chunks, vecs
}

func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	var b [20]byte
	p := len(b)
	for i > 0 {
		p--
		b[p] = byte('0' + i%10)
		i /= 10
	}
	return string(b[p:])
}

func TestAddSearchPersist(t *testing.T) {
	dir := t.TempDir()
	s, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	chunks, vecs := makeChunks(200)
	if err := s.AddChunks(chunks, vecs); err != nil {
		t.Fatal(err)
	}
	s.MarkIndexed("/docs/file.txt", time.Unix(1000, 0))
	if s.ChunkCount() != 200 {
		t.Fatalf("chunk count = %d", s.ChunkCount())
	}
	if s.FileCount() != 1 {
		t.Fatalf("file count = %d", s.FileCount())
	}

	// Query before save.
	gotChunks, scores, err := s.Search(vecs[50], 5)
	if err != nil {
		t.Fatal(err)
	}
	if len(gotChunks) == 0 || gotChunks[0].ID != "chunk-50" {
		t.Fatalf("expected chunk-50 as top result, got %+v", gotChunks)
	}
	if scores[0] < 0.99 {
		t.Fatalf("self-match score too low: %f", scores[0])
	}

	if err := s.Save(); err != nil {
		t.Fatal(err)
	}

	// Reopen and verify chunks + search survive the round trip.
	s2, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	if s2.ChunkCount() != 200 {
		t.Fatalf("reopened chunk count = %d", s2.ChunkCount())
	}
	if s2.FileCount() != 1 {
		t.Fatalf("reopened file count = %d", s2.FileCount())
	}
	gotChunks2, _, err := s2.Search(vecs[50], 5)
	if err != nil {
		t.Fatal(err)
	}
	if len(gotChunks2) == 0 || gotChunks2[0].ID != "chunk-50" {
		t.Fatalf("after reload expected chunk-50, got %+v", gotChunks2)
	}
	if gotChunks2[0].Text != "chunk text number 50" {
		t.Fatalf("chunk text not preserved: %q", gotChunks2[0].Text)
	}
}

func TestHasFileChangeDetection(t *testing.T) {
	dir := t.TempDir()
	s, _ := Open(dir)
	mt := time.Unix(5000, 0)
	if s.HasFile("/a.txt", mt) {
		t.Fatal("should not have file before indexing")
	}
	s.MarkIndexed("/a.txt", mt)
	if !s.HasFile("/a.txt", mt) {
		t.Fatal("should have file at same mtime")
	}
	if s.HasFile("/a.txt", mt.Add(time.Second)) {
		t.Fatal("should detect changed mtime")
	}

	// Persisted manifest must survive reopen.
	s.Save()
	s2, _ := Open(dir)
	if !s2.HasFile("/a.txt", mt) {
		t.Fatal("manifest not persisted")
	}
}
