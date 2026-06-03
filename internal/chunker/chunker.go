// Package chunker splits text into overlapping, semantically-aware chunks.
//
// The strategy favors natural boundaries: paragraphs first, then sentences,
// and only as a last resort a hard character split. Each chunk carries a few
// characters of overlap from the previous chunk so that retrieval does not
// lose context that straddles a boundary.
package chunker

import (
	"crypto/sha256"
	"encoding/hex"
	"strings"
)

// Chunk is a single retrievable unit of text plus provenance metadata.
type Chunk struct {
	ID      string // sha256(content[:64] + source); deterministic
	Source  string // file path the chunk came from
	PageNum int    // 1-based page for PDFs; 0 for non-paginated sources
	Start   int    // byte offset of the chunk body within the source
	End     int    // byte offset (exclusive) of the chunk body within the source
	Text    string // chunk text, including any prepended overlap
}

// Config controls chunk sizing. Zero values fall back to sensible defaults.
type Config struct {
	ChunkSize    int // target characters per chunk
	Overlap      int // characters carried forward from the previous chunk
	MinChunkSize int // chunks smaller than this are discarded (after merging)
}

func (c Config) withDefaults() Config {
	if c.ChunkSize <= 0 {
		c.ChunkSize = 512
	}
	if c.Overlap < 0 {
		c.Overlap = 0
	}
	if c.Overlap == 0 {
		c.Overlap = 64
	}
	if c.MinChunkSize <= 0 {
		c.MinChunkSize = 50
	}
	// Overlap must be smaller than the chunk size or hard-splitting never
	// makes forward progress.
	if c.Overlap >= c.ChunkSize {
		c.Overlap = c.ChunkSize / 4
	}
	return c
}

// Split breaks text into chunks. PageNum on every returned chunk is 0; callers
// that have per-page text (PDFs) should chunk each page separately and set
// PageNum themselves.
//
// Note: the design doc named this function Chunk, but Go forbids a function and
// a type sharing a name in one package, so it is Split here.
func Split(text, source string, cfg Config) []Chunk {
	cfg = cfg.withDefaults()

	// segments are the raw building blocks: paragraphs, further split into
	// sentence-sized pieces or hard-split pieces when oversized. Each segment
	// retains its byte offset within the original text.
	segments := buildSegments(text, cfg)

	// Merge adjacent segments greedily up to ChunkSize so we don't emit a
	// flurry of tiny chunks for short paragraphs.
	merged := mergeSegments(segments, cfg)

	return assemble(merged, source, cfg)
}

type segment struct {
	text  string
	start int
	end   int
}

// buildSegments produces paragraph-level segments, recursively splitting any
// that exceed ChunkSize on sentence then hard boundaries.
func buildSegments(text string, cfg Config) []segment {
	var out []segment
	for _, p := range splitParagraphs(text) {
		if len(p.text) <= cfg.ChunkSize {
			out = append(out, p)
			continue
		}
		for _, s := range splitSentences(p, cfg) {
			if len(s.text) <= cfg.ChunkSize {
				out = append(out, s)
				continue
			}
			out = append(out, hardSplit(s, cfg)...)
		}
	}
	return out
}

// splitParagraphs splits on blank lines (one or more), preserving byte offsets.
func splitParagraphs(text string) []segment {
	var out []segment
	i := 0
	n := len(text)
	for i < n {
		// Skip leading whitespace/blank separators between paragraphs.
		for i < n && isParaBreak(text, i) {
			i++
		}
		if i >= n {
			break
		}
		start := i
		// Advance until we hit a double newline (paragraph break).
		for i < n {
			if text[i] == '\n' && i+1 < n && nextIsNewline(text, i+1) {
				break
			}
			i++
		}
		body := strings.TrimSpace(text[start:i])
		if body != "" {
			// Recompute trimmed offsets so Start/End point at real content.
			ts := start + indexOfTrimmedStart(text[start:i])
			out = append(out, segment{text: body, start: ts, end: ts + len(body)})
		}
	}
	return out
}

func isParaBreak(text string, i int) bool {
	c := text[i]
	return c == '\n' || c == '\r' || c == ' ' || c == '\t'
}

// nextIsNewline reports whether position j begins another newline (handling
// \n\n and \r\n\r\n style breaks loosely).
func nextIsNewline(text string, j int) bool {
	for j < len(text) {
		switch text[j] {
		case '\n':
			return true
		case '\r', ' ', '\t':
			j++
		default:
			return false
		}
	}
	return false
}

func indexOfTrimmedStart(s string) int {
	for i := 0; i < len(s); i++ {
		switch s[i] {
		case ' ', '\t', '\n', '\r':
		default:
			return i
		}
	}
	return 0
}

// splitSentences breaks an oversized paragraph on sentence terminators,
// accumulating sentences until adding another would exceed ChunkSize.
func splitSentences(p segment, cfg Config) []segment {
	bounds := sentenceBoundaries(p.text)
	var out []segment
	var cur strings.Builder
	curStart := p.start
	flush := func(end int) {
		body := strings.TrimSpace(cur.String())
		if body != "" {
			ts := curStart + indexOfTrimmedStart(cur.String())
			out = append(out, segment{text: body, start: ts, end: ts + len(body)})
		}
		cur.Reset()
	}
	prev := 0
	for _, b := range bounds {
		sentence := p.text[prev:b]
		if cur.Len() > 0 && cur.Len()+len(sentence) > cfg.ChunkSize {
			flush(p.start + prev)
			curStart = p.start + prev
		}
		if cur.Len() == 0 {
			curStart = p.start + prev
		}
		cur.WriteString(sentence)
		prev = b
	}
	if prev < len(p.text) {
		if cur.Len() == 0 {
			curStart = p.start + prev
		}
		cur.WriteString(p.text[prev:])
	}
	flush(p.end)
	return out
}

// sentenceBoundaries returns indices just past each sentence terminator.
func sentenceBoundaries(s string) []int {
	var bounds []int
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c == '.' || c == '?' || c == '!' {
			// Treat the terminator plus trailing space as the boundary.
			j := i + 1
			if j < len(s) && (s[j] == ' ' || s[j] == '\n') {
				bounds = append(bounds, j+1)
				i = j
			}
		}
	}
	return bounds
}

// hardSplit chops an oversized sentence into ChunkSize windows with overlap
// carried forward at the character level.
func hardSplit(s segment, cfg Config) []segment {
	var out []segment
	runes := s.text
	step := cfg.ChunkSize - cfg.Overlap
	if step <= 0 {
		step = cfg.ChunkSize
	}
	for i := 0; i < len(runes); i += step {
		end := i + cfg.ChunkSize
		if end > len(runes) {
			end = len(runes)
		}
		out = append(out, segment{
			text:  runes[i:end],
			start: s.start + i,
			end:   s.start + end,
		})
		if end == len(runes) {
			break
		}
	}
	return out
}

// mergeSegments greedily combines adjacent segments until reaching ChunkSize so
// that tiny paragraphs become reasonably sized chunks. Hard-split segments are
// already near ChunkSize and merge naturally.
func mergeSegments(segs []segment, cfg Config) []segment {
	var out []segment
	i := 0
	for i < len(segs) {
		cur := segs[i]
		j := i + 1
		for j < len(segs) {
			// +2 accounts for the "\n\n" joiner between merged paragraphs.
			if len(cur.text)+len(segs[j].text)+2 > cfg.ChunkSize {
				break
			}
			cur.text = cur.text + "\n\n" + segs[j].text
			cur.end = segs[j].end
			j++
		}
		out = append(out, cur)
		i = j
	}
	return out
}

// assemble turns merged segments into Chunks, prepending overlap from the prior
// chunk's tail and discarding anything below MinChunkSize.
func assemble(segs []segment, source string, cfg Config) []Chunk {
	var chunks []Chunk
	var prevTail string
	for _, s := range segs {
		body := s.text
		if len(body) < cfg.MinChunkSize {
			// Too small to stand alone and merging already happened; skip.
			continue
		}
		text := body
		if prevTail != "" {
			text = prevTail + "\n\n" + body
		}
		chunks = append(chunks, Chunk{
			ID:      chunkID(text, source),
			Source:  source,
			PageNum: 0,
			Start:   s.start,
			End:     s.end,
			Text:    text,
		})
		prevTail = tail(body, cfg.Overlap)
	}
	return chunks
}

// tail returns the last n characters of s (byte-wise; inputs are UTF-8 text and
// we accept the rare boundary split since overlap is only a retrieval aid).
func tail(s string, n int) string {
	if n <= 0 || len(s) <= n {
		return s
	}
	return s[len(s)-n:]
}

// chunkID is deterministic: identical (text, source) always yields the same ID,
// which is what makes manifest-based dedup on re-index reliable.
func chunkID(text, source string) string {
	h := sha256.New()
	body := text
	if len(body) > 64 {
		body = body[:64]
	}
	h.Write([]byte(body))
	h.Write([]byte{0})
	h.Write([]byte(source))
	return hex.EncodeToString(h.Sum(nil))[:32]
}
