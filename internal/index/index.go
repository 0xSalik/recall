// Package index provides approximate and exact nearest-neighbor search over
// embedding vectors. Two implementations share one interface: FlatIndex (exact
// brute-force cosine) and HNSW (graph-based approximate search).
//
// All vectors are assumed to be L2-normalized, so cosine similarity is just the
// dot product. Higher scores mean more similar.
package index

// SearchResult is one hit from a nearest-neighbor query.
type SearchResult struct {
	ID       string
	Score    float32 // cosine similarity (dot product of unit vectors)
	ChunkIdx int     // caller-supplied index into a parallel chunk array
}

// Entry is a stored vector with its id and chunk index. Entries lets callers
// extract the full contents of an index so they can rebuild a filtered copy —
// which is how deletion is implemented (neither index type supports in-place
// removal, so the store rebuilds from the surviving entries).
type Entry struct {
	ID       string
	Vec      []float32
	ChunkIdx int
}

// Index is the common contract for vector indexes.
type Index interface {
	// Add inserts a vector with an associated id and chunk index.
	Add(id string, vec []float32, chunkIdx int) error
	// Search returns up to k results ordered by descending score.
	Search(query []float32, k int) ([]SearchResult, error)
	// Len reports the number of indexed vectors.
	Len() int
	// Entries returns every stored vector (order unspecified).
	Entries() []Entry
	// Save serializes the index to a file at path.
	Save(path string) error
	// Load replaces the index contents from a file at path.
	Load(path string) error
}

// dot computes the dot product of two equal-length vectors.
func dot(a, b []float32) float32 {
	var s float32
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}
