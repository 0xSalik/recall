package cmd

import (
	"flag"
	"fmt"
	"time"

	"github.com/0xSalik/recall/internal/rag"
)

// Index implements `recall index <path> [<path2> ...]`.
func Index(args []string) {
	fs := flag.NewFlagSet("index", flag.ExitOnError)
	store := fs.String("store", defaultStoreDir(), "path to store directory")
	embedModel := fs.String("embed", defaultEmbedModel, "embedding model path")
	genModel := fs.String("gen", defaultGenModel, "generation model path")
	llama := fs.String("llama", "", "path to llama-cli binary (default: search PATH)")
	ext := fs.String("ext", "", "comma-separated extensions to include (default: all supported)")
	fs.Parse(args)

	paths := fs.Args()
	if len(paths) == 0 {
		fail("index requires at least one path")
	}

	r, err := rag.New(*store, *embedModel, *genModel, *llama)
	if err != nil {
		fail("%v", err)
	}
	r.SetExtensions(splitExts(*ext))

	start := time.Now()
	results, err := r.Index(paths)
	if err != nil {
		fail("%v", err)
	}

	total := len(results)
	skipped := 0
	chunks := 0
	idx := 0
	for _, res := range results {
		if res.Skipped {
			skipped++
			continue
		}
		idx++
		chunks += res.Chunks
		if res.Pages > 0 {
			fmt.Printf("  [%d/%d] %s (%d chunks, %d pages)\n", idx, total-skipped, res.Path, res.Chunks, res.Pages)
		} else {
			fmt.Printf("  [%d/%d] %s (%d chunks)\n", idx, total-skipped, res.Path, res.Chunks)
		}
	}
	if skipped > 0 {
		fmt.Printf("  Skipped %d files (unchanged)\n", skipped)
	}
	fmt.Printf("Done. %d chunks indexed in %.1fs\n", chunks, time.Since(start).Seconds())
}
