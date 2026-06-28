package cmd

import (
	"flag"
	"fmt"

	"github.com/0xSalik/recall/internal/rag"
)

// Refresh implements `recall refresh [paths...]`: prune indexed files that no
// longer exist, re-index files that changed, and (if paths are given) discover
// and index new files under them.
func Refresh(args []string) {
	fs := flag.NewFlagSet("refresh", flag.ExitOnError)
	store := fs.String("store", defaultStoreDir(), "path to store directory")
	embedModel := fs.String("embed", "", modelFlagHelp)
	bin := fs.String("bin", "", "directory containing llama.cpp binaries (prepended to PATH)")
	paths := parseArgs(fs, args)
	resolveEngine(*bin)

	r, err := rag.NewIndexer(*store, resolveEmbedModel(*embedModel))
	if err != nil {
		fail("%v", err)
	}
	res, err := r.Refresh(paths)
	if err != nil {
		fail("%v", err)
	}

	for _, p := range res.Deleted {
		fmt.Printf("  pruned (deleted on disk) %s\n", p)
	}
	indexed, skipped := 0, 0
	for _, ir := range res.Reindexed {
		if ir.Skipped {
			skipped++
			continue
		}
		indexed++
		fmt.Printf("  reindexed %s (%d chunks)\n", ir.Path, ir.Chunks)
	}
	fmt.Printf("\nRefresh complete: %d pruned, %d indexed, %d unchanged.\n",
		len(res.Deleted), indexed, skipped)
}
