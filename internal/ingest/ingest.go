// Package ingest reads files from disk and turns them into Documents: raw text
// plus enough metadata (format, per-page content, mod time) for the rest of the
// pipeline to chunk and attribute them correctly.
package ingest

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// Document is the extracted, chunk-ready representation of a single file.
type Document struct {
	Path     string
	Format   string   // "pdf", "markdown", "text", "code"
	Content  string   // full extracted text
	Pages    []string // PDF only: text per page (1-based when read)
	Modified time.Time
}

// codeExtensions are treated as source code (function-aware chunking upstream).
var codeExtensions = map[string]bool{
	".go": true, ".py": true, ".js": true, ".ts": true, ".rs": true,
	".cpp": true, ".c": true, ".h": true, ".java": true, ".rb": true, ".sh": true,
}

// markdownExtensions get the markdown stripper.
var markdownExtensions = map[string]bool{
	".md": true, ".markdown": true, ".mdown": true,
}

// textExtensions are passed through verbatim.
var textExtensions = map[string]bool{
	".txt": true, ".text": true, ".log": true, ".rst": true, "": true,
}

// formatFor returns the ingestion format for a path based on its extension.
func formatFor(path string) string {
	ext := strings.ToLower(filepath.Ext(path))
	switch {
	case ext == ".pdf":
		return "pdf"
	case markdownExtensions[ext]:
		return "markdown"
	case codeExtensions[ext]:
		return "code"
	case textExtensions[ext]:
		return "text"
	default:
		return ""
	}
}

// IngestFile reads and extracts a single file. The format is chosen by
// extension; unknown extensions are treated as plain text if they look textual.
func IngestFile(path string) (Document, error) {
	info, err := os.Stat(path)
	if err != nil {
		return Document{}, err
	}
	format := formatFor(path)

	switch format {
	case "pdf":
		return ingestPDF(path, info.ModTime())
	case "markdown":
		return ingestMarkdown(path, info.ModTime())
	case "code":
		return ingestCode(path, info.ModTime())
	case "text":
		return ingestPlainText(path, info.ModTime())
	default:
		// Unknown extension: only ingest if it appears to be text.
		data, err := os.ReadFile(path)
		if err != nil {
			return Document{}, err
		}
		if isBinary(data) {
			return Document{}, fmt.Errorf("ingest: %s appears to be binary, skipping", path)
		}
		return Document{
			Path:     path,
			Format:   "text",
			Content:  string(data),
			Modified: info.ModTime(),
		}, nil
	}
}

// IngestDir walks root recursively and ingests every supported file. If
// extensions is non-empty, only files with those extensions (e.g. ".md") are
// considered. Directories like .git and node_modules and binary files are
// skipped silently.
func IngestDir(root string, extensions []string) ([]Document, error) {
	allow := make(map[string]bool, len(extensions))
	for _, e := range extensions {
		e = strings.ToLower(strings.TrimSpace(e))
		if e == "" {
			continue
		}
		if !strings.HasPrefix(e, ".") {
			e = "." + e
		}
		allow[e] = true
	}

	var docs []Document
	err := filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			// Unreadable entry: skip it but keep walking.
			return nil
		}
		if d.IsDir() {
			if skipDir(d.Name()) {
				return filepath.SkipDir
			}
			return nil
		}
		if d.Name() == ".DS_Store" {
			return nil
		}
		ext := strings.ToLower(filepath.Ext(path))
		if len(allow) > 0 && !allow[ext] {
			return nil
		}
		if len(allow) == 0 && formatFor(path) == "" {
			// No explicit filter and unsupported extension: skip.
			return nil
		}
		doc, ierr := IngestFile(path)
		if ierr != nil {
			// Log-and-skip: a single bad file shouldn't abort the whole walk.
			fmt.Fprintf(os.Stderr, "warning: skipping %s: %v\n", path, ierr)
			return nil
		}
		if strings.TrimSpace(doc.Content) == "" {
			return nil
		}
		docs = append(docs, doc)
		return nil
	})
	return docs, err
}

func skipDir(name string) bool {
	switch name {
	case ".git", "node_modules", ".svn", ".hg", "vendor", "__pycache__", ".venv", "dist", "build":
		return true
	}
	return false
}

// isBinary reports whether data looks binary by scanning the first 512 bytes
// for a NUL byte.
func isBinary(data []byte) bool {
	n := len(data)
	if n > 512 {
		n = 512
	}
	for i := 0; i < n; i++ {
		if data[i] == 0 {
			return true
		}
	}
	return false
}
