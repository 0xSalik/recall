//go:build !bundle

package bootstrap

import "io"

// This file provides the no-op asset source used by ordinary builds (the
// `go build` everyone runs during development). No models or binaries are
// embedded, so recall falls back to the on-disk cache, downloads, and PATH.
//
// The "bundle" build tag swaps in assets_bundle.go, which serves real files from
// an embedded filesystem populated by CI. See docs/RELEASE.md.

func embeddedOpen(string) (io.ReadCloser, bool) { return nil, false }

func embeddedAvailable() bool { return false }
