package cmd

import (
	"flag"
	"fmt"

	"github.com/0xSalik/recall/internal/rag"
)

// List implements `recall list`: print the indexed files and their chunk counts.
func List(args []string) {
	fs := flag.NewFlagSet("list", flag.ExitOnError)
	store := fs.String("store", defaultStoreDir(), "path to store directory")
	fs.Parse(args)

	r, err := rag.NewManager(*store)
	if err != nil {
		fail("%v", err)
	}
	files := r.ListFiles()
	if len(files) == 0 {
		fmt.Println("No files indexed.")
		return
	}
	total := 0
	for _, f := range files {
		fmt.Printf("  %6d  %s\n", f.Chunks, f.Path)
		total += f.Chunks
	}
	fmt.Printf("\n%d files, %d chunks\n", len(files), total)
}
