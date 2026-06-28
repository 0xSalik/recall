package cmd

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/0xSalik/recall/internal/rag"
)

// Clear implements `recall clear`: remove all indexed chunks. Prompts for
// confirmation unless --yes is given.
func Clear(args []string) {
	fs := flag.NewFlagSet("clear", flag.ExitOnError)
	store := fs.String("store", defaultStoreDir(), "path to store directory")
	yes := fs.Bool("yes", false, "skip the confirmation prompt")
	fs.Parse(args)

	r, err := rag.NewManager(*store)
	if err != nil {
		fail("%v", err)
	}
	files := r.ListFiles()
	if len(files) == 0 {
		fmt.Println("Index is already empty.")
		return
	}
	if !*yes && !confirm(fmt.Sprintf("Clear the entire index (%d files)? [y/N] ", len(files))) {
		fmt.Println("Aborted.")
		return
	}
	if err := r.Clear(); err != nil {
		fail("%v", err)
	}
	fmt.Printf("Cleared %d files from the index.\n", len(files))
}

// confirm reads a yes/no answer from stdin.
func confirm(prompt string) bool {
	fmt.Print(prompt)
	reader := bufio.NewReader(os.Stdin)
	line, _ := reader.ReadString('\n')
	line = strings.ToLower(strings.TrimSpace(line))
	return line == "y" || line == "yes"
}
