// Package web embeds the single-file browser UI so the serve command can ship
// it without any external assets or build step. The UI is plain HTML + vanilla
// JS; it talks to the local server over fetch (JSON) and SSE (streaming).
package web

import _ "embed"

// IndexHTML is the served UI document.
//
//go:embed index.html
var IndexHTML []byte
