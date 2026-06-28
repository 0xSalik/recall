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

func makeChunksFor(source string, n, start int) ([]chunker.Chunk, [][]float32) {
	r := rand.New(rand.NewSource(int64(start + 1)))
	chunks := make([]chunker.Chunk, n)
	vecs := make([][]float32, n)
	for i := range chunks {
		chunks[i] = chunker.Chunk{
			ID:     source + "-" + itoa(start+i),
			Source: source,
			Text:   "text " + itoa(start+i),
		}
		vecs[i] = unitVec(r, 48)
	}
	return chunks, vecs
}

func TestRemoveByPrefixAndClear(t *testing.T) {
	s, _ := Open(t.TempDir())
	ca, va := makeChunksFor("/docs/a/one.txt", 5, 0)
	cb, vb := makeChunksFor("/docs/a/two.txt", 3, 100)
	cc, vc := makeChunksFor("/docs/b/three.txt", 4, 200)
	s.AddChunks(ca, va)
	s.MarkIndexed("/docs/a/one.txt", time.Unix(1, 0))
	s.AddChunks(cb, vb)
	s.MarkIndexed("/docs/a/two.txt", time.Unix(2, 0))
	s.AddChunks(cc, vc)
	s.MarkIndexed("/docs/b/three.txt", time.Unix(3, 0))

	if s.ChunkCount() != 12 || s.FileCount() != 3 {
		t.Fatalf("setup wrong: %d chunks, %d files", s.ChunkCount(), s.FileCount())
	}

	// Remove the /docs/a directory: should drop 8 chunks and 2 files.
	n, files, err := s.Remove("/docs/a")
	if err != nil {
		t.Fatal(err)
	}
	if n != 8 {
		t.Fatalf("removed %d chunks, want 8", n)
	}
	if len(files) != 2 {
		t.Fatalf("removed %d files, want 2", len(files))
	}
	if s.ChunkCount() != 4 || s.FileCount() != 1 {
		t.Fatalf("after remove: %d chunks, %d files", s.ChunkCount(), s.FileCount())
	}

	// Surviving chunks must still be searchable and rebuilt correctly.
	got, _, err := s.Search(vc[0], 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) == 0 || got[0].Source != "/docs/b/three.txt" {
		t.Fatalf("search after remove returned wrong chunk: %+v", got)
	}

	// Persistence round-trip after removal.
	dir := s.Dir()
	if err := s.Save(); err != nil {
		t.Fatal(err)
	}
	s2, _ := Open(dir)
	if s2.ChunkCount() != 4 || s2.FileCount() != 1 {
		t.Fatalf("reopened after remove: %d chunks, %d files", s2.ChunkCount(), s2.FileCount())
	}

	// Clear wipes everything.
	s2.Clear()
	if s2.ChunkCount() != 0 || s2.FileCount() != 0 {
		t.Fatalf("after clear: %d chunks, %d files", s2.ChunkCount(), s2.FileCount())
	}
}

func TestRemoveFilesExact(t *testing.T) {
	s, _ := Open(t.TempDir())
	ca, va := makeChunksFor("/x/a.txt", 3, 0)
	cb, vb := makeChunksFor("/x/ab.txt", 2, 50) // shares the /x/a prefix but is a different file
	s.AddChunks(ca, va)
	s.MarkIndexed("/x/a.txt", time.Unix(1, 0))
	s.AddChunks(cb, vb)
	s.MarkIndexed("/x/ab.txt", time.Unix(2, 0))

	// Exact-file removal must not touch the similarly-named file.
	n, err := s.RemoveFiles([]string{"/x/a.txt"})
	if err != nil {
		t.Fatal(err)
	}
	if n != 3 {
		t.Fatalf("removed %d, want 3", n)
	}
	if !s.IsIndexed("/x/ab.txt") || s.IsIndexed("/x/a.txt") {
		t.Fatal("RemoveFiles affected the wrong file")
	}
	if s.ChunkCount() != 2 {
		t.Fatalf("chunk count = %d, want 2", s.ChunkCount())
	}
}

func TestListFiles(t *testing.T) {
	s, _ := Open(t.TempDir())
	ca, va := makeChunksFor("/z/b.txt", 2, 0)
	cb, vb := makeChunksFor("/z/a.txt", 3, 10)
	s.AddChunks(ca, va)
	s.MarkIndexed("/z/b.txt", time.Unix(1, 0))
	s.AddChunks(cb, vb)
	s.MarkIndexed("/z/a.txt", time.Unix(2, 0))

	files := s.ListFiles()
	if len(files) != 2 {
		t.Fatalf("listed %d files, want 2", len(files))
	}
	// Sorted by path: a.txt before b.txt.
	if files[0].Path != "/z/a.txt" || files[0].Chunks != 3 {
		t.Fatalf("unexpected first file: %+v", files[0])
	}
	if files[1].Path != "/z/b.txt" || files[1].Chunks != 2 {
		t.Fatalf("unexpected second file: %+v", files[1])
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
