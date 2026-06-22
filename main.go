// Command recall is a local-first retrieval-augmented generation tool. It
// indexes your files into a vector store and answers questions over them using
// local llama.cpp models — no cloud services involved.
package main

import (
	"fmt"
	"os"

	"github.com/0xSalik/recall/cmd"
)

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}
	switch os.Args[1] {
	case "index":
		cmd.Index(os.Args[2:])
	case "query":
		cmd.Query(os.Args[2:])
	case "status":
		cmd.Status(os.Args[2:])
	case "serve":
		cmd.Serve(os.Args[2:])
	case "help", "-h", "--help":
		usage()
	default:
		fmt.Fprintf(os.Stderr, "recall: unknown command %q\n\n", os.Args[1])
		usage()
		os.Exit(2)
	}
}

func usage() {
	fmt.Fprint(os.Stderr, `recall — local-first RAG over your files

Usage:
  recall index <path> [<path> ...]   ingest and index files or directories
  recall query "<question>"          ask a question over the indexed corpus
  recall status                      show store statistics
  recall serve                       start the local HTTP UI + JSON/SSE API

Run "recall <command> -h" for command-specific flags.
`)
}
