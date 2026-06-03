package chunker

import (
	"strings"
	"testing"
)

func TestSingleShortParagraph(t *testing.T) {
	text := "This is a single short paragraph that should become exactly one chunk because it fits well within the configured chunk size."
	chunks := Split(text, "notes.txt", Config{})
	if len(chunks) != 1 {
		t.Fatalf("expected 1 chunk, got %d", len(chunks))
	}
	if chunks[0].Text != text {
		t.Fatalf("chunk text altered: %q", chunks[0].Text)
	}
	if chunks[0].Source != "notes.txt" {
		t.Fatalf("source not set: %q", chunks[0].Source)
	}
	if chunks[0].ID == "" {
		t.Fatal("ID not populated")
	}
}

func TestLongParagraphSplitsOnSentences(t *testing.T) {
	// Build a paragraph well over the chunk size from distinct sentences.
	var b strings.Builder
	for i := 0; i < 40; i++ {
		b.WriteString("The quick brown fox jumps over the lazy dog. ")
	}
	chunks := Split(b.String(), "x.txt", Config{ChunkSize: 200, Overlap: 20})
	if len(chunks) < 2 {
		t.Fatalf("expected multiple chunks for long paragraph, got %d", len(chunks))
	}
	for i, c := range chunks {
		// Allow a little slack for prepended overlap.
		if len(c.Text) > 200+40 {
			t.Fatalf("chunk %d exceeds size budget: %d", i, len(c.Text))
		}
	}
}

func TestOverlapCarryForward(t *testing.T) {
	var b strings.Builder
	for i := 0; i < 30; i++ {
		b.WriteString("Sentence number is here now. ")
	}
	overlap := 24
	chunks := Split(b.String(), "x.txt", Config{ChunkSize: 150, Overlap: overlap})
	if len(chunks) < 2 {
		t.Fatalf("need >=2 chunks to test overlap, got %d", len(chunks))
	}
	for i := 1; i < len(chunks); i++ {
		prefix := chunks[i].Text
		// chunk[i] must start with the last `overlap` chars of chunk[i-1]'s body.
		prevBody := chunks[i-1].Text
		expected := prevBody[len(prevBody)-overlap:]
		if !strings.HasPrefix(prefix, expected) {
			t.Fatalf("chunk %d does not start with overlap of previous.\nexpected prefix: %q\ngot start: %q",
				i, expected, prefix[:min(len(prefix), overlap+10)])
		}
	}
}

func TestMinChunkSizeDiscard(t *testing.T) {
	text := "tiny.\n\n" + strings.Repeat("word ", 200)
	chunks := Split(text, "x.txt", Config{ChunkSize: 300, Overlap: 20, MinChunkSize: 50})
	for _, c := range chunks {
		// Body (excluding overlap prefix) is what's gated; total should still
		// comfortably exceed min for all emitted chunks.
		if len(c.Text) < 50 {
			t.Fatalf("emitted chunk below MinChunkSize: %q", c.Text)
		}
	}
}

func TestDeterministicID(t *testing.T) {
	text := strings.Repeat("hello world. ", 50)
	a := Split(text, "same.txt", Config{})
	b := Split(text, "same.txt", Config{})
	if len(a) != len(b) {
		t.Fatalf("nondeterministic chunk count: %d vs %d", len(a), len(b))
	}
	for i := range a {
		if a[i].ID != b[i].ID {
			t.Fatalf("nondeterministic ID at %d: %s vs %s", i, a[i].ID, b[i].ID)
		}
	}
	// Different source must change the ID.
	c := Split(text, "other.txt", Config{})
	if c[0].ID == a[0].ID {
		t.Fatal("ID should depend on source")
	}
}

func TestEmptyInput(t *testing.T) {
	if got := Split("", "x.txt", Config{}); len(got) != 0 {
		t.Fatalf("expected no chunks for empty input, got %d", len(got))
	}
	if got := Split("   \n\n   \t  ", "x.txt", Config{}); len(got) != 0 {
		t.Fatalf("expected no chunks for whitespace input, got %d", len(got))
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
