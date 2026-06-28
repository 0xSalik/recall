// Package cmd implements the recall subcommands. Each command parses its own
// flags with the standard library flag package (no third-party CLI library) and
// is dispatched from main.
package cmd

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/0xSalik/recall/internal/bootstrap"
)

// modelFlagHelp documents the model flags, which now default to empty: an empty
// value means "resolve automatically" (cache -> embedded -> download).
const modelFlagHelp = "model path (default: managed automatically and fetched on first use)"

// defaultStoreDir is ~/.recall, falling back to ./.recall if the home dir is
// unavailable.
func defaultStoreDir() string {
	return bootstrap.Home()
}

// resolveEngine makes the llama.cpp binaries discoverable: it prefers an
// embedded/cached copy (bundle builds), then the user's --bin override, then the
// ambient PATH. The override is applied last so it takes precedence.
func resolveEngine(binFlag string) {
	if dir, err := bootstrap.EnsureEngine(); err == nil && dir != "" {
		addBinToPath(dir)
	}
	addBinToPath(binFlag)
}

// resolveEmbedModel returns a usable embedding-model path, downloading it (with
// a progress bar) if necessary. An empty flagVal means "managed automatically".
func resolveEmbedModel(flagVal string) string {
	return resolveModel("embedding model", flagVal, bootstrap.EnsureEmbedModel)
}

// resolveGenModel returns a usable generation-model path. With no override and
// no cached/embedded copy this triggers the large first-run download.
func resolveGenModel(flagVal string) string {
	return resolveModel("generation model", flagVal, bootstrap.EnsureGenModel)
}

func resolveModel(label, flagVal string, ensure func(string, bootstrap.ProgressFunc) (string, error)) string {
	progress, done := bootstrap.CLIProgress(os.Stderr, "  "+label)
	path, err := ensure(flagVal, progress)
	done()
	if err != nil {
		fail("%v", err)
	}
	return path
}

// splitExts parses a comma-separated extension list (e.g. ".md,.pdf,go") into a
// normalized slice, or nil when empty.
func splitExts(csv string) []string {
	csv = strings.TrimSpace(csv)
	if csv == "" {
		return nil
	}
	parts := strings.Split(csv, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		if !strings.HasPrefix(p, ".") {
			p = "." + p
		}
		out = append(out, p)
	}
	return out
}

// addBinToPath prepends dir to PATH so the llama.cpp binaries are discoverable
// without the user editing their shell environment. dir is made absolute first,
// since a relative PATH entry only resolves from one working directory.
func addBinToPath(dir string) {
	if dir == "" {
		return
	}
	abs, err := filepath.Abs(dir)
	if err != nil {
		abs = dir
	}
	os.Setenv("PATH", abs+string(os.PathListSeparator)+os.Getenv("PATH"))
}

// parseArgs parses flags that may be interspersed with positional arguments.
// The standard flag package stops at the first non-flag token, which would make
// `recall index ~/docs --ext .md` ignore --ext. This repeatedly parses, peeling
// off one positional at a time, so flags before or after paths both work.
func parseArgs(fs *flag.FlagSet, args []string) []string {
	var positionals []string
	for {
		if err := fs.Parse(args); err != nil {
			os.Exit(2)
		}
		rest := fs.Args()
		if len(rest) == 0 {
			break
		}
		positionals = append(positionals, rest[0])
		args = rest[1:]
	}
	return positionals
}

// fail prints an error to stderr and exits non-zero.
func fail(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "recall: "+format+"\n", args...)
	os.Exit(1)
}

// humanBytes renders a byte count as a human-readable string.
func humanBytes(n int64) string {
	const unit = 1024
	if n < unit {
		return fmt.Sprintf("%d B", n)
	}
	div, exp := int64(unit), 0
	for x := n / unit; x >= unit; x /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %cB", float64(n)/float64(div), "KMGTPE"[exp])
}
