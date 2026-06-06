package ingest

import (
	"os"
	"strings"
	"time"
)

// ingestMarkdown strips markdown syntax down to readable plain text while
// preserving fenced code blocks verbatim (they carry semantic value) and
// keeping heading text (useful context). This is a deliberately small state
// machine rather than a full markdown parser.
func ingestMarkdown(path string, mod time.Time) (Document, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return Document{}, err
	}
	content := stripMarkdown(string(data))
	return Document{
		Path:     path,
		Format:   "markdown",
		Content:  content,
		Modified: mod,
	}, nil
}

func stripMarkdown(src string) string {
	lines := strings.Split(src, "\n")
	var out []string
	inFence := false
	var fence string // the fence marker that opened the current block

	for _, line := range lines {
		trimmed := strings.TrimSpace(line)

		// Fenced code blocks: ``` or ~~~. Preserve content verbatim.
		if !inFence && (strings.HasPrefix(trimmed, "```") || strings.HasPrefix(trimmed, "~~~")) {
			inFence = true
			fence = trimmed[:3]
			continue // drop the opening fence line itself
		}
		if inFence {
			if strings.HasPrefix(trimmed, fence) {
				inFence = false
				continue // drop the closing fence line
			}
			out = append(out, line) // keep code line untouched
			continue
		}

		out = append(out, stripInline(stripBlockPrefix(line)))
	}
	return strings.Join(out, "\n")
}

// stripBlockPrefix removes leading block-level markers: heading hashes,
// blockquote markers, list bullets, and horizontal rules.
func stripBlockPrefix(line string) string {
	t := strings.TrimRight(line, " \t")
	ls := strings.TrimLeft(t, " \t")

	// Horizontal rule -> blank.
	if ls == "---" || ls == "***" || ls == "___" {
		return ""
	}

	// ATX headings: strip leading #'s, keep the text.
	if strings.HasPrefix(ls, "#") {
		ls = strings.TrimLeft(ls, "#")
		return strings.TrimSpace(ls)
	}

	// Blockquote.
	for strings.HasPrefix(ls, ">") {
		ls = strings.TrimSpace(ls[1:])
	}

	// Unordered list bullets.
	if strings.HasPrefix(ls, "- ") || strings.HasPrefix(ls, "* ") || strings.HasPrefix(ls, "+ ") {
		ls = strings.TrimSpace(ls[2:])
	}

	return ls
}

// stripInline removes inline emphasis, inline code backticks, image and link
// syntax while keeping the visible text.
func stripInline(line string) string {
	// Links and images: ![alt](url) and [text](url) -> alt / text.
	line = unlink(line)

	var b strings.Builder
	for i := 0; i < len(line); i++ {
		c := line[i]
		switch c {
		case '*', '_', '`':
			// Drop emphasis / inline code markers; keep the wrapped text.
			continue
		default:
			b.WriteByte(c)
		}
	}
	return b.String()
}

// unlink replaces [text](url) with text and ![alt](url) with alt.
func unlink(s string) string {
	var b strings.Builder
	i := 0
	for i < len(s) {
		// Image: ![alt](url)
		if s[i] == '!' && i+1 < len(s) && s[i+1] == '[' {
			i++ // skip '!', fall through to link handling for the [
		}
		if s[i] == '[' {
			closeBracket := strings.IndexByte(s[i:], ']')
			if closeBracket > 0 {
				j := i + closeBracket
				// Must be immediately followed by "(...)" to be a link.
				if j+1 < len(s) && s[j+1] == '(' {
					closeParen := strings.IndexByte(s[j+1:], ')')
					if closeParen > 0 {
						b.WriteString(s[i+1 : j]) // the link/alt text
						i = j + 1 + closeParen + 1
						continue
					}
				}
			}
		}
		b.WriteByte(s[i])
		i++
	}
	return b.String()
}
