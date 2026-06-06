package ingest

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func writeFile(t *testing.T, dir, name, content string) string {
	t.Helper()
	p := filepath.Join(dir, name)
	if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestIngestPlainText(t *testing.T) {
	dir := t.TempDir()
	p := writeFile(t, dir, "notes.txt", "hello world\nsecond line")
	doc, err := IngestFile(p)
	if err != nil {
		t.Fatal(err)
	}
	if doc.Format != "text" {
		t.Fatalf("format = %q, want text", doc.Format)
	}
	if doc.Content != "hello world\nsecond line" {
		t.Fatalf("content mismatch: %q", doc.Content)
	}
	if doc.Modified.IsZero() {
		t.Fatal("modified time not set")
	}
}

func TestIngestMarkdown(t *testing.T) {
	dir := t.TempDir()
	md := "# Title\n\nSome **bold** and _italic_ and `code` text.\n\n" +
		"[a link](http://example.com) here.\n\n" +
		"```go\nfunc main() {}\n```\n"
	p := writeFile(t, dir, "doc.md", md)
	doc, err := IngestFile(p)
	if err != nil {
		t.Fatal(err)
	}
	if doc.Format != "markdown" {
		t.Fatalf("format = %q", doc.Format)
	}
	if strings.Contains(doc.Content, "**") || strings.Contains(doc.Content, "`code`") {
		t.Fatalf("inline markdown not stripped: %q", doc.Content)
	}
	if !strings.Contains(doc.Content, "Title") {
		t.Fatal("heading text should be preserved")
	}
	if !strings.Contains(doc.Content, "bold") || !strings.Contains(doc.Content, "italic") {
		t.Fatal("emphasized text should be preserved")
	}
	if !strings.Contains(doc.Content, "a link") {
		t.Fatalf("link text should be preserved: %q", doc.Content)
	}
	if strings.Contains(doc.Content, "http://example.com") {
		t.Fatal("link URL should be removed")
	}
	if !strings.Contains(doc.Content, "func main() {}") {
		t.Fatal("fenced code block content should be preserved verbatim")
	}
}

func TestIngestCode(t *testing.T) {
	dir := t.TempDir()
	code := "package main\nfunc a() {}\nfunc b() {}\n"
	p := writeFile(t, dir, "x.go", code)
	doc, err := IngestFile(p)
	if err != nil {
		t.Fatal(err)
	}
	if doc.Format != "code" {
		t.Fatalf("format = %q", doc.Format)
	}
	if !strings.HasPrefix(doc.Content, "// File: ") {
		t.Fatalf("code should be prefixed with file header: %q", doc.Content[:20])
	}
	if !strings.Contains(doc.Content, "func a()") || !strings.Contains(doc.Content, "func b()") {
		t.Fatal("code body missing")
	}
}

func TestIngestBinarySkipped(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "blob.dat")
	if err := os.WriteFile(p, []byte{0x00, 0x01, 0x02, 0x00}, 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := IngestFile(p); err == nil {
		t.Fatal("expected binary file to be rejected")
	}
}

func TestIngestDir(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, dir, "a.txt", "plain text content here")
	writeFile(t, dir, "b.md", "# Heading\n\nmarkdown body content")
	writeFile(t, dir, "c.go", "package x\nfunc f() { return }")
	// Should be skipped (excluded directory):
	nm := filepath.Join(dir, "node_modules")
	os.MkdirAll(nm, 0o755)
	writeFile(t, nm, "pkg.txt", "should not be ingested")
	os.WriteFile(filepath.Join(dir, "bin.dat"), []byte{0, 1, 2}, 0o644)

	docs, err := IngestDir(dir, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(docs) != 3 {
		t.Fatalf("expected 3 docs, got %d", len(docs))
	}
	for _, d := range docs {
		if strings.TrimSpace(d.Content) == "" {
			t.Fatalf("empty content for %s", d.Path)
		}
	}
}

func TestIngestDirExtensionFilter(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, dir, "a.txt", "plain text")
	writeFile(t, dir, "b.md", "# md content")
	writeFile(t, dir, "c.go", "package x")

	docs, err := IngestDir(dir, []string{".md", "go"})
	if err != nil {
		t.Fatal(err)
	}
	if len(docs) != 2 {
		t.Fatalf("expected 2 docs with filter, got %d", len(docs))
	}
}
