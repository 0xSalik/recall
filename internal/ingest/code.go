package ingest

import (
	"os"
	"path/filepath"
	"strings"
	"time"
)

// ingestCode reads a source file and lays it out so that chunking later happens
// at function/class boundaries where possible. We keep the code intact
// (comments and docstrings included — they're valuable) and prepend a
// "// File: <path>" header so retrieved snippets are self-identifying.
//
// The Content we return is the file with logical units separated by blank-line
// boundaries that the chunker's paragraph splitter will respect. We do not
// rewrite the code; we only ensure unit boundaries are visible as blank lines.
func ingestCode(path string, mod time.Time) (Document, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return Document{}, err
	}
	ext := strings.ToLower(filepath.Ext(path))
	body := splitUnits(string(data), ext)

	header := "// File: " + path + "\n\n"
	return Document{
		Path:     path,
		Format:   "code",
		Content:  header + body,
		Modified: mod,
	}, nil
}

// splitUnits inserts a blank line before each top-level function/class boundary
// so downstream paragraph chunking aligns with code structure. Boundaries are
// language-specific; everything else falls back to existing blank-line blocks.
func splitUnits(src, ext string) string {
	lines := strings.Split(src, "\n")
	var out []string

	isBoundary := boundaryDetector(ext)

	for i, line := range lines {
		if i > 0 && isBoundary(line) {
			// Ensure exactly one blank line precedes a new unit.
			if len(out) > 0 && strings.TrimSpace(out[len(out)-1]) != "" {
				out = append(out, "")
			}
		}
		out = append(out, line)
	}
	return strings.Join(out, "\n")
}

// boundaryDetector returns a predicate that reports whether a line begins a new
// top-level code unit for the given extension.
func boundaryDetector(ext string) func(string) bool {
	switch ext {
	case ".go":
		return func(l string) bool {
			return strings.HasPrefix(l, "func ")
		}
	case ".py":
		return func(l string) bool {
			// Top-level def/class (no leading indentation).
			return strings.HasPrefix(l, "def ") || strings.HasPrefix(l, "class ")
		}
	case ".rs":
		return func(l string) bool {
			t := strings.TrimSpace(l)
			return strings.HasPrefix(t, "fn ") || strings.HasPrefix(t, "pub fn ") ||
				strings.HasPrefix(t, "impl ") || strings.HasPrefix(t, "struct ")
		}
	case ".js", ".ts":
		return func(l string) bool {
			t := strings.TrimSpace(l)
			return strings.HasPrefix(t, "function ") || strings.HasPrefix(t, "export function ") ||
				strings.HasPrefix(t, "class ") || strings.HasPrefix(t, "export class ")
		}
	case ".java", ".cpp", ".c", ".h":
		return func(l string) bool {
			t := strings.TrimSpace(l)
			// Heuristic: class/struct declarations are clear boundaries.
			return strings.HasPrefix(t, "class ") || strings.HasPrefix(t, "struct ")
		}
	case ".rb":
		return func(l string) bool {
			t := strings.TrimSpace(l)
			return strings.HasPrefix(t, "def ") || strings.HasPrefix(t, "class ") ||
				strings.HasPrefix(t, "module ")
		}
	default:
		// .sh and unknown: rely on existing blank-line blocks only.
		return func(string) bool { return false }
	}
}
