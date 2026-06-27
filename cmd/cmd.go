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
)

// Defaults for flags shared across commands.
const (
	defaultEmbedModel = "models/nomic-embed-text-v1.5.Q4_K_M.gguf"
	defaultGenModel   = "models/phi-3-mini-4k-instruct.Q4_K_M.gguf"
)

// defaultStoreDir is ~/.recall, falling back to ./.recall if the home dir is
// unavailable.
func defaultStoreDir() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return ".recall"
	}
	return filepath.Join(home, ".recall")
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
