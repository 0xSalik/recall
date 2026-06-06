package ingest

import (
	"os"
	"time"
)

// ingestPlainText reads a file verbatim. No transformation is applied.
func ingestPlainText(path string, mod time.Time) (Document, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return Document{}, err
	}
	return Document{
		Path:     path,
		Format:   "text",
		Content:  string(data),
		Modified: mod,
	}, nil
}
