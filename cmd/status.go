package cmd

import (
	"flag"
	"fmt"
	"strings"

	"github.com/0xSalik/recall/internal/store"
)

// Status implements `recall status`. It reads only the store (no models), so it
// works without llama.cpp installed.
func Status(args []string) {
	fs := flag.NewFlagSet("status", flag.ExitOnError)
	dir := fs.String("store", defaultStoreDir(), "path to store directory")
	fs.Parse(args)

	s, err := store.Open(*dir)
	if err != nil {
		fail("%v", err)
	}

	fmt.Printf("Store: %s\n", s.Dir())
	fmt.Printf("Files indexed: %d\n", s.FileCount())
	fmt.Printf("Chunks: %d\n", s.ChunkCount())
	fmt.Printf("Index type: %s\n", strings.ToUpper(s.IndexType()))
	fmt.Printf("Store size: %s\n", humanBytes(s.DiskSize()))
}
