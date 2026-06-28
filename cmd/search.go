package cmd

import (
	"flag"
	"fmt"
	"strings"

	"github.com/0xSalik/recall/internal/rag"
)

// Search implements `recall search <query>`: retrieval only. It embeds the query
// and prints the top matching chunks with scores and sources, without running
// the generation model. Fast and useful for inspecting what the index returns.
func Search(args []string) {
	fs := flag.NewFlagSet("search", flag.ExitOnError)
	store := fs.String("store", defaultStoreDir(), "path to store directory")
	embedModel := fs.String("embed", defaultEmbedModel, "embedding model path")
	bin := fs.String("bin", "", "directory containing llama.cpp binaries (prepended to PATH)")
	topK := fs.Int("k", 5, "number of chunks to return")
	rest := parseArgs(fs, args)
	if len(rest) == 0 {
		fail("search requires a query")
	}
	addBinToPath(*bin)
	query := strings.Join(rest, " ")

	r, err := rag.NewIndexer(*store, *embedModel)
	if err != nil {
		fail("%v", err)
	}
	sources, err := r.Retrieve(query, *topK)
	if err != nil {
		fail("%v", err)
	}
	if len(sources) == 0 {
		fmt.Println("No matches.")
		return
	}
	for i, s := range sources {
		loc := s.Chunk.Source
		if s.Chunk.PageNum > 0 {
			loc = fmt.Sprintf("%s (p.%d)", loc, s.Chunk.PageNum)
		}
		fmt.Printf("%d. [%.3f] %s\n", i+1, s.Score, loc)
		fmt.Printf("   %s\n\n", snippet(s.Chunk.Text, 280))
	}
}

// snippet collapses whitespace and truncates text to n runes for display.
func snippet(text string, n int) string {
	text = strings.Join(strings.Fields(text), " ")
	r := []rune(text)
	if len(r) <= n {
		return text
	}
	return string(r[:n]) + "…"
}
