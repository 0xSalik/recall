//go:build bundle

package bootstrap

import (
	"embed"
	"io"
)

// This file is compiled only for release "bundle" builds (go build -tags bundle).
// CI populates internal/bootstrap/assets/ with the platform engine binaries
// (assets/bin/llama-embedding, ...) and the embedding model
// (assets/models/<embed>.gguf) before building, so they get baked into the
// executable. The large generation model is intentionally NOT embedded; it is
// downloaded on first use to keep the release asset under GitHub's 2 GB limit.
//
// The all: prefix ensures files whose names start with "." or "_" are included
// too, and embedding a directory tolerates an otherwise-empty tree (the
// committed assets/.gitkeep), so this builds even before CI drops real assets in.

//go:embed all:assets
var assetsFS embed.FS

func embeddedOpen(rel string) (io.ReadCloser, bool) {
	f, err := assetsFS.Open("assets/" + rel)
	if err != nil {
		return nil, false
	}
	return f, true
}

func embeddedAvailable() bool { return true }
