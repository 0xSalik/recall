package ingest

import (
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/ledongthuc/pdf"
)

// ingestPDF extracts text from a PDF page by page. Pages that fail extraction
// (common with scanned, image-only PDFs) are logged and skipped rather than
// aborting the whole document. The full Content concatenates pages with
// "[Page N]" markers so chunks can be attributed to a page.
func ingestPDF(path string, mod time.Time) (Document, error) {
	f, r, err := pdf.Open(path)
	if err != nil {
		return Document{}, fmt.Errorf("open pdf: %w", err)
	}
	defer f.Close()

	numPages := r.NumPage()
	pages := make([]string, 0, numPages)
	var full strings.Builder

	for i := 1; i <= numPages; i++ {
		page := r.Page(i)
		if page.V.IsNull() {
			pages = append(pages, "")
			continue
		}
		text, perr := page.GetPlainText(nil)
		if perr != nil {
			fmt.Fprintf(os.Stderr, "warning: %s page %d: extraction failed: %v\n", path, i, perr)
			pages = append(pages, "")
			continue
		}
		text = strings.TrimSpace(text)
		pages = append(pages, text)
		if text != "" {
			fmt.Fprintf(&full, "\n\n[Page %d]\n\n%s", i, text)
		}
	}

	content := strings.TrimSpace(full.String())
	if content == "" {
		// Likely a scanned/image-only PDF: surface a warning, return empty so
		// the caller skips it instead of indexing nothing.
		fmt.Fprintf(os.Stderr, "warning: %s yielded no extractable text (scanned PDF?)\n", path)
	}

	return Document{
		Path:     path,
		Format:   "pdf",
		Content:  content,
		Pages:    pages,
		Modified: mod,
	}, nil
}
