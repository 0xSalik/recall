package index

import (
	"math"
	"math/rand"
	"path/filepath"
	"testing"
	"time"
)

func randomUnitVec(r *rand.Rand, dims int) []float32 {
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

func buildVecs(n, dims int, seed int64) [][]float32 {
	r := rand.New(rand.NewSource(seed))
	vecs := make([][]float32, n)
	for i := range vecs {
		vecs[i] = randomUnitVec(r, dims)
	}
	return vecs
}

func fmtID(i int) string { return "id-" + itoa(i) }

func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	var b [20]byte
	pos := len(b)
	for i > 0 {
		pos--
		b[pos] = byte('0' + i%10)
		i /= 10
	}
	return string(b[pos:])
}

func TestFlatExactNearest(t *testing.T) {
	dims := 64
	vecs := buildVecs(1000, dims, 42)
	f := NewFlatIndex(dims)
	for i, v := range vecs {
		if err := f.Add(fmtID(i), v, i); err != nil {
			t.Fatal(err)
		}
	}
	// Querying with an exact copy of vector 500 must return it as top-1.
	res, err := f.Search(vecs[500], 5)
	if err != nil {
		t.Fatal(err)
	}
	if res[0].ID != fmtID(500) {
		t.Fatalf("flat top-1 = %s, want id-500", res[0].ID)
	}
	if math.Abs(float64(res[0].Score)-1.0) > 1e-4 {
		t.Fatalf("self-similarity = %f, want ~1.0", res[0].Score)
	}
}

func TestHNSWRecallVsFlat(t *testing.T) {
	dims := 64
	n := 2000
	vecs := buildVecs(n, dims, 7)

	flat := NewFlatIndex(dims)
	h := newHNSWSeeded(dims, 99)
	for i, v := range vecs {
		flat.Add(fmtID(i), v, i)
		if err := h.Add(fmtID(i), v, i); err != nil {
			t.Fatal(err)
		}
	}

	r := rand.New(rand.NewSource(1234))
	queries := 200
	matches := 0
	for q := 0; q < queries; q++ {
		query := randomUnitVec(r, dims)
		fres, _ := flat.Search(query, 1)
		hres, _ := h.Search(query, 1)
		if len(hres) > 0 && len(fres) > 0 && hres[0].ID == fres[0].ID {
			matches++
		}
	}
	recall := float64(matches) / float64(queries)
	if recall < 0.95 {
		t.Fatalf("HNSW top-1 recall = %.3f, want >= 0.95", recall)
	}
	t.Logf("HNSW top-1 recall = %.3f", recall)
}

func TestHNSWFasterThanFlat(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping speed test in -short mode")
	}
	dims := 128
	n := 10000
	vecs := buildVecs(n, dims, 21)

	flat := NewFlatIndex(dims)
	h := newHNSWSeeded(dims, 5)
	for i, v := range vecs {
		flat.Add(fmtID(i), v, i)
		h.Add(fmtID(i), v, i)
	}

	r := rand.New(rand.NewSource(55))
	queries := make([][]float32, 200)
	for i := range queries {
		queries[i] = randomUnitVec(r, dims)
	}

	start := time.Now()
	for _, q := range queries {
		flat.Search(q, 5)
	}
	flatDur := time.Since(start)

	start = time.Now()
	for _, q := range queries {
		h.Search(q, 5)
	}
	hnswDur := time.Since(start)

	t.Logf("flat=%v hnsw=%v speedup=%.1fx", flatDur, hnswDur, float64(flatDur)/float64(hnswDur))
	if float64(flatDur)/float64(hnswDur) < 5.0 {
		t.Fatalf("HNSW only %.1fx faster than flat, want >= 5x", float64(flatDur)/float64(hnswDur))
	}
}

func TestFlatSaveLoad(t *testing.T) {
	dims := 32
	vecs := buildVecs(100, dims, 3)
	f := NewFlatIndex(dims)
	for i, v := range vecs {
		f.Add(fmtID(i), v, i*2)
	}
	path := filepath.Join(t.TempDir(), "flat.bin")
	if err := f.Save(path); err != nil {
		t.Fatal(err)
	}
	g := NewFlatIndex(0)
	if err := g.Load(path); err != nil {
		t.Fatal(err)
	}
	if g.Len() != f.Len() {
		t.Fatalf("len mismatch: %d vs %d", g.Len(), f.Len())
	}
	r1, _ := f.Search(vecs[10], 5)
	r2, _ := g.Search(vecs[10], 5)
	for i := range r1 {
		if r1[i].ID != r2[i].ID || r1[i].ChunkIdx != r2[i].ChunkIdx {
			t.Fatalf("result %d mismatch after reload", i)
		}
	}
}

func TestHNSWSaveLoad(t *testing.T) {
	dims := 32
	vecs := buildVecs(500, dims, 11)
	h := newHNSWSeeded(dims, 8)
	for i, v := range vecs {
		h.Add(fmtID(i), v, i)
	}
	path := filepath.Join(t.TempDir(), "hnsw.gob")
	if err := h.Save(path); err != nil {
		t.Fatal(err)
	}
	g := NewHNSW(0)
	if err := g.Load(path); err != nil {
		t.Fatal(err)
	}
	if g.Len() != h.Len() {
		t.Fatalf("len mismatch: %d vs %d", g.Len(), h.Len())
	}
	r1, _ := h.Search(vecs[42], 5)
	r2, _ := g.Search(vecs[42], 5)
	if len(r1) != len(r2) || r1[0].ID != r2[0].ID {
		t.Fatalf("search results differ after reload: %v vs %v", r1, r2)
	}
}

func TestCosineSimilarityHelper(t *testing.T) {
	a := []float32{1, 0, 0}
	b := []float32{1, 0, 0}
	if math.Abs(float64(cosineSimilarity(a, b))-1.0) > 1e-6 {
		t.Fatal("identical vectors should have cosine 1.0")
	}
	c := []float32{0, 1, 0}
	if math.Abs(float64(cosineSimilarity(a, c))) > 1e-6 {
		t.Fatal("orthogonal vectors should have cosine 0")
	}
}
