package cmd

import (
	"context"
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/0xSalik/recall/internal/rag"
)

// Query implements `recall query "<question>"`.
func Query(args []string) {
	fs := flag.NewFlagSet("query", flag.ExitOnError)
	store := fs.String("store", defaultStoreDir(), "path to store directory")
	embedModel := fs.String("embed", defaultEmbedModel, "embedding model path")
	genModel := fs.String("gen", defaultGenModel, "generation model path")
	llama := fs.String("llama", "", "path to generation binary (default: search PATH)")
	bin := fs.String("bin", "", "directory containing llama.cpp binaries (prepended to PATH)")
	k := fs.Int("k", 5, "number of chunks to retrieve")
	stream := fs.Bool("stream", false, "stream output token by token")
	sources := fs.Bool("sources", false, "print source files after the answer")
	question := strings.Join(parseArgs(fs, args), " ")
	if question == "" {
		fail("query requires a question")
	}
	addBinToPath(*bin)

	r, err := rag.New(*store, *embedModel, *genModel, *llama)
	if err != nil {
		fail("%v", err)
	}

	var answer rag.Answer
	if *stream {
		fmt.Print("Answer: ")
		tokens := make(chan string, 32)
		done := make(chan struct{})
		go func() {
			for tok := range tokens {
				fmt.Print(tok)
			}
			close(done)
		}()
		answer, err = r.AskStream(context.Background(), question, *k, tokens)
		close(tokens)
		<-done
		fmt.Println()
	} else {
		answer, err = r.Ask(question, *k)
		if err == nil {
			fmt.Printf("Answer: %s\n", answer.Text)
		}
	}
	if err != nil {
		fail("%v", err)
	}

	if *sources {
		fmt.Fprintln(os.Stdout, "\nSources:")
		for _, s := range answer.Sources {
			if s.Chunk.PageNum > 0 {
				fmt.Printf("  %s, page %d (score: %.2f)\n", s.Chunk.Source, s.Chunk.PageNum, s.Score)
			} else {
				fmt.Printf("  %s (score: %.2f)\n", s.Chunk.Source, s.Score)
			}
		}
	}
}
