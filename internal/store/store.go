// Package store persists chunks and their vector index to disk so that a file
// is only ingested and embedded once. The on-disk layout under the store
// directory is:
//
//	chunks.json    all Chunk structs, in index order
//	index.bin      serialized vector index (HNSW by default)
//	manifest.json  index metadata + {path: modtime} for change detection
package store

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/0xSalik/recall/internal/chunker"
	"github.com/0xSalik/recall/internal/index"
)

const (
	chunksFile   = "chunks.json"
	indexFile    = "index.bin"
	manifestFile = "manifest.json"
)

// Store is the persistent home for an indexed corpus. The chunks slice is
// parallel to the vector index: index ChunkIdx i refers to chunks[i].
type Store struct {
	dir    string
	index  index.Index
	chunks []chunker.Chunk
	man    manifest
}

// manifest records enough to rebuild the index handle and to detect which files
// have already been indexed (and at what mtime).
type manifest struct {
	IndexType string               `json:"index_type"` // "hnsw" or "flat"
	Dims      int                  `json:"dims"`
	Files     map[string]time.Time `json:"files"`
}

// Open loads an existing store from dir, creating an empty one (HNSW index) if
// none exists yet.
func Open(dir string) (*Store, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}
	s := &Store{
		dir: dir,
		man: manifest{IndexType: "hnsw", Files: map[string]time.Time{}},
	}

	// Load manifest if present.
	if data, err := os.ReadFile(filepath.Join(dir, manifestFile)); err == nil {
		if err := json.Unmarshal(data, &s.man); err != nil {
			return nil, fmt.Errorf("store: corrupt manifest: %w", err)
		}
		if s.man.Files == nil {
			s.man.Files = map[string]time.Time{}
		}
	}

	// Instantiate the right index type, then load it if a file exists.
	s.index = newIndex(s.man.IndexType, s.man.Dims)
	if _, err := os.Stat(filepath.Join(dir, indexFile)); err == nil {
		if err := s.index.Load(filepath.Join(dir, indexFile)); err != nil {
			return nil, fmt.Errorf("store: loading index: %w", err)
		}
	}

	// Load chunks.
	if data, err := os.ReadFile(filepath.Join(dir, chunksFile)); err == nil {
		if err := json.Unmarshal(data, &s.chunks); err != nil {
			return nil, fmt.Errorf("store: corrupt chunks: %w", err)
		}
	}
	return s, nil
}

func newIndex(kind string, dims int) index.Index {
	switch kind {
	case "flat":
		return index.NewFlatIndex(dims)
	default:
		return index.NewHNSW(dims)
	}
}

// IndexType reports the configured index type ("hnsw" or "flat").
func (s *Store) IndexType() string { return s.man.IndexType }

// AddChunks appends chunks and their vectors to the store. len(chunks) must
// equal len(vecs). The vector index ChunkIdx is set to the chunk's position in
// the store's chunk slice so Search can map results back to chunks.
func (s *Store) AddChunks(chunks []chunker.Chunk, vecs [][]float32) error {
	if len(chunks) != len(vecs) {
		return fmt.Errorf("store: %d chunks but %d vectors", len(chunks), len(vecs))
	}
	for i, c := range chunks {
		idx := len(s.chunks)
		if err := s.index.Add(c.ID, vecs[i], idx); err != nil {
			return err
		}
		s.chunks = append(s.chunks, c)
		if s.man.Dims == 0 && len(vecs[i]) > 0 {
			s.man.Dims = len(vecs[i])
		}
	}
	return nil
}

// MarkIndexed records that path was indexed at modTime. This is what makes
// re-indexing idempotent: a later HasFile with the same mtime returns true.
//
// (The design doc folded this into AddChunks, but Chunk carries no mtime, so it
// is a separate call the ingest pipeline makes per file.)
func (s *Store) MarkIndexed(path string, modTime time.Time) {
	s.man.Files[path] = modTime
}

// HasFile reports whether path was already indexed at the given modTime.
func (s *Store) HasFile(path string, modTime time.Time) bool {
	mt, ok := s.man.Files[path]
	return ok && mt.Equal(modTime)
}

// IsIndexed reports whether path is recorded in the manifest at any modtime.
func (s *Store) IsIndexed(path string) bool {
	_, ok := s.man.Files[path]
	return ok
}

// FileInfo describes an indexed file for listing.
type FileInfo struct {
	Path    string    `json:"path"`
	ModTime time.Time `json:"modtime"`
	Chunks  int       `json:"chunks"`
}

// ListFiles returns the indexed files with per-file chunk counts, sorted by
// path.
func (s *Store) ListFiles() []FileInfo {
	counts := make(map[string]int, len(s.man.Files))
	for _, c := range s.chunks {
		counts[c.Source]++
	}
	out := make([]FileInfo, 0, len(s.man.Files))
	for p, mt := range s.man.Files {
		out = append(out, FileInfo{Path: p, ModTime: mt, Chunks: counts[p]})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Path < out[j].Path })
	return out
}

// Clear removes all chunks, vectors, and manifest entries, resetting the store
// to empty. Call Save to persist.
func (s *Store) Clear() {
	s.chunks = nil
	s.man.Files = map[string]time.Time{}
	s.man.Dims = 0
	s.index = newIndex(s.man.IndexType, 0)
}

// Remove deletes everything indexed at exactly target, or (treating target as a
// directory) under target. It returns the number of chunks removed and the list
// of files removed. Call Save to persist.
func (s *Store) Remove(target string) (int, []string, error) {
	t := normalize(target)
	match := func(p string) bool {
		np := normalize(p)
		return np == t || strings.HasPrefix(np, t+string(filepath.Separator))
	}
	var removedFiles []string
	for p := range s.man.Files {
		if match(p) {
			removedFiles = append(removedFiles, p)
		}
	}
	n, err := s.removeWhere(func(c chunker.Chunk) bool { return match(c.Source) })
	if err != nil {
		return 0, nil, err
	}
	for _, p := range removedFiles {
		delete(s.man.Files, p)
	}
	sort.Strings(removedFiles)
	return n, removedFiles, nil
}

// RemoveFiles deletes all chunks belonging to the exact file paths given and
// drops them from the manifest, rebuilding the index once. Call Save to persist.
func (s *Store) RemoveFiles(paths []string) (int, error) {
	if len(paths) == 0 {
		return 0, nil
	}
	set := make(map[string]bool, len(paths))
	for _, p := range paths {
		set[normalize(p)] = true
	}
	n, err := s.removeWhere(func(c chunker.Chunk) bool { return set[normalize(c.Source)] })
	if err != nil {
		return 0, err
	}
	for p := range s.man.Files {
		if set[normalize(p)] {
			delete(s.man.Files, p)
		}
	}
	return n, nil
}

// removeWhere drops chunks matching pred and rebuilds the index from the
// survivors' vectors (extracted from the current index). It does not touch the
// manifest; callers do that.
func (s *Store) removeWhere(pred func(chunker.Chunk) bool) (int, error) {
	vecByIdx := make(map[int][]float32, len(s.chunks))
	for _, e := range s.index.Entries() {
		vecByIdx[e.ChunkIdx] = e.Vec
	}
	keptChunks := make([]chunker.Chunk, 0, len(s.chunks))
	keptVecs := make([][]float32, 0, len(s.chunks))
	removed := 0
	for i, c := range s.chunks {
		if pred(c) {
			removed++
			continue
		}
		keptChunks = append(keptChunks, c)
		keptVecs = append(keptVecs, vecByIdx[i])
	}
	s.chunks = keptChunks
	if err := s.rebuildIndex(keptVecs); err != nil {
		return removed, err
	}
	return removed, nil
}

// rebuildIndex constructs a fresh index from the current chunks and the given
// parallel vectors (chunk i uses vecs[i]).
func (s *Store) rebuildIndex(vecs [][]float32) error {
	idx := newIndex(s.man.IndexType, s.man.Dims)
	for i := range s.chunks {
		if vecs[i] == nil {
			return fmt.Errorf("store: missing vector for chunk %d during rebuild", i)
		}
		if err := idx.Add(s.chunks[i].ID, vecs[i], i); err != nil {
			return err
		}
	}
	s.index = idx
	return nil
}

// normalize cleans a path for stable comparison.
func normalize(p string) string {
	return filepath.Clean(p)
}

// Search embeds-free: it takes an already-embedded query vector and returns the
// top-k chunks with their scores.
func (s *Store) Search(queryVec []float32, k int) ([]chunker.Chunk, []float32, error) {
	results, err := s.index.Search(queryVec, k)
	if err != nil {
		return nil, nil, err
	}
	chunks := make([]chunker.Chunk, 0, len(results))
	scores := make([]float32, 0, len(results))
	for _, r := range results {
		if r.ChunkIdx < 0 || r.ChunkIdx >= len(s.chunks) {
			continue
		}
		chunks = append(chunks, s.chunks[r.ChunkIdx])
		scores = append(scores, r.Score)
	}
	return chunks, scores, nil
}

// FileCount returns the number of distinct indexed files.
func (s *Store) FileCount() int { return len(s.man.Files) }

// ChunkCount returns the number of indexed chunks.
func (s *Store) ChunkCount() int { return len(s.chunks) }

// Dir returns the store's directory.
func (s *Store) Dir() string { return s.dir }

// DiskSize returns the total size of the store's files on disk in bytes.
func (s *Store) DiskSize() int64 {
	var total int64
	for _, name := range []string{chunksFile, indexFile, manifestFile} {
		if info, err := os.Stat(filepath.Join(s.dir, name)); err == nil {
			total += info.Size()
		}
	}
	return total
}

// Save persists chunks, index, and manifest atomically-ish (each file written
// then renamed into place).
func (s *Store) Save() error {
	chunkData, err := json.Marshal(s.chunks)
	if err != nil {
		return err
	}
	if err := writeAtomic(filepath.Join(s.dir, chunksFile), chunkData); err != nil {
		return err
	}
	if err := s.index.Save(filepath.Join(s.dir, indexFile)); err != nil {
		return err
	}
	manData, err := json.MarshalIndent(s.man, "", "  ")
	if err != nil {
		return err
	}
	return writeAtomic(filepath.Join(s.dir, manifestFile), manData)
}

// writeAtomic writes data to a temp file then renames it over the target so a
// crash mid-write can't truncate an existing good file.
func writeAtomic(path string, data []byte) error {
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}
