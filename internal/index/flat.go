package index

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"sort"
)

// FlatIndex is an exact nearest-neighbor index: every query scans every vector.
// It is O(n) per search but always correct, which makes it the ground-truth
// baseline for HNSW recall tests and perfectly adequate for small corpora
// (< ~10k chunks).
type FlatIndex struct {
	dims  int
	ids   []string
	chunk []int
	vecs  [][]float32
}

// NewFlatIndex creates an empty flat index. dims may be 0; it is inferred from
// the first vector added.
func NewFlatIndex(dims int) *FlatIndex {
	return &FlatIndex{dims: dims}
}

func (f *FlatIndex) Add(id string, vec []float32, chunkIdx int) error {
	if f.dims == 0 {
		f.dims = len(vec)
	}
	if len(vec) != f.dims {
		return fmt.Errorf("flat: vector dim %d != index dim %d", len(vec), f.dims)
	}
	cp := make([]float32, len(vec))
	copy(cp, vec)
	f.ids = append(f.ids, id)
	f.chunk = append(f.chunk, chunkIdx)
	f.vecs = append(f.vecs, cp)
	return nil
}

func (f *FlatIndex) Len() int { return len(f.ids) }

func (f *FlatIndex) Entries() []Entry {
	out := make([]Entry, len(f.ids))
	for i := range f.ids {
		out[i] = Entry{ID: f.ids[i], Vec: f.vecs[i], ChunkIdx: f.chunk[i]}
	}
	return out
}

func (f *FlatIndex) Search(query []float32, k int) ([]SearchResult, error) {
	if len(query) != f.dims && f.dims != 0 {
		return nil, fmt.Errorf("flat: query dim %d != index dim %d", len(query), f.dims)
	}
	results := make([]SearchResult, len(f.vecs))
	for i, v := range f.vecs {
		results[i] = SearchResult{ID: f.ids[i], Score: dot(query, v), ChunkIdx: f.chunk[i]}
	}
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})
	if k < len(results) {
		results = results[:k]
	}
	return results, nil
}

// Save writes a compact binary format:
//
//	int32 dims
//	int32 count
//	for each vector: int32 chunkIdx, int32 idLen, id bytes, dims*float32
func (f *FlatIndex) Save(path string) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()
	w := bufio.NewWriter(file)
	defer w.Flush()

	if err := binary.Write(w, binary.LittleEndian, int32(f.dims)); err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, int32(len(f.vecs))); err != nil {
		return err
	}
	for i, v := range f.vecs {
		if err := binary.Write(w, binary.LittleEndian, int32(f.chunk[i])); err != nil {
			return err
		}
		id := []byte(f.ids[i])
		if err := binary.Write(w, binary.LittleEndian, int32(len(id))); err != nil {
			return err
		}
		if _, err := w.Write(id); err != nil {
			return err
		}
		if err := binary.Write(w, binary.LittleEndian, v); err != nil {
			return err
		}
	}
	return nil
}

func (f *FlatIndex) Load(path string) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()
	r := bufio.NewReader(file)

	var dims, count int32
	if err := binary.Read(r, binary.LittleEndian, &dims); err != nil {
		return err
	}
	if err := binary.Read(r, binary.LittleEndian, &count); err != nil {
		return err
	}
	f.dims = int(dims)
	f.ids = make([]string, count)
	f.chunk = make([]int, count)
	f.vecs = make([][]float32, count)
	for i := int32(0); i < count; i++ {
		var chunkIdx, idLen int32
		if err := binary.Read(r, binary.LittleEndian, &chunkIdx); err != nil {
			return err
		}
		if err := binary.Read(r, binary.LittleEndian, &idLen); err != nil {
			return err
		}
		if idLen < 0 || idLen > 1<<20 {
			return fmt.Errorf("flat: corrupt id length %d", idLen)
		}
		idBuf := make([]byte, idLen)
		if _, err := readFull(r, idBuf); err != nil {
			return err
		}
		vec := make([]float32, dims)
		if err := binary.Read(r, binary.LittleEndian, vec); err != nil {
			return err
		}
		f.ids[i] = string(idBuf)
		f.chunk[i] = int(chunkIdx)
		f.vecs[i] = vec
	}
	return nil
}

func readFull(r *bufio.Reader, buf []byte) (int, error) {
	total := 0
	for total < len(buf) {
		n, err := r.Read(buf[total:])
		total += n
		if err != nil {
			return total, err
		}
	}
	return total, nil
}

// cosineSimilarity is a helper for tests and callers working with raw vectors.
func cosineSimilarity(a, b []float32) float32 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return float32(dot / (math.Sqrt(na) * math.Sqrt(nb)))
}
