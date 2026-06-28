package cmd

import (
	"flag"
	"fmt"
	"path/filepath"

	"github.com/0xSalik/recall/internal/rag"
)

// Remove implements `recall remove <path>`: drop everything indexed at an exact
// file path or, treating the path as a directory, everything beneath it.
func Remove(args []string) {
	fs := flag.NewFlagSet("remove", flag.ExitOnError)
	store := fs.String("store", defaultStoreDir(), "path to store directory")
	targets := parseArgs(fs, args)
	if len(targets) == 0 {
		fail("remove requires at least one path")
	}

	r, err := rag.NewManager(*store)
	if err != nil {
		fail("%v", err)
	}
	totalChunks, totalFiles := 0, 0
	for _, t := range targets {
		abs, aerr := filepath.Abs(t)
		if aerr != nil {
			abs = t
		}
		n, files, rerr := r.Remove(abs)
		if rerr != nil {
			fail("%v", rerr)
		}
		if len(files) == 0 {
			fmt.Printf("Nothing indexed under %s\n", t)
			continue
		}
		for _, f := range files {
			fmt.Printf("  removed %s\n", f)
		}
		totalChunks += n
		totalFiles += len(files)
	}
	fmt.Printf("Removed %d files (%d chunks).\n", totalFiles, totalChunks)
}
