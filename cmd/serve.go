package cmd

import (
	"encoding/json"
	"flag"
	"fmt"
	"net/http"

	"github.com/0xSalik/recall/internal/rag"
)

// Serve implements `recall serve`: a local HTTP server exposing a JSON API over
// the RAG pipeline.
func Serve(args []string) {
	fs := flag.NewFlagSet("serve", flag.ExitOnError)
	addr := fs.String("addr", "localhost:8080", "listen address")
	store := fs.String("store", defaultStoreDir(), "path to store directory")
	embedModel := fs.String("embed", defaultEmbedModel, "embedding model path")
	genModel := fs.String("gen", defaultGenModel, "generation model path")
	llama := fs.String("llama", "", "path to llama-cli binary (default: search PATH)")
	fs.Parse(args)

	r, err := rag.New(*store, *embedModel, *genModel, *llama)
	if err != nil {
		fail("%v", err)
	}

	srv := newServer(r)
	mux := http.NewServeMux()
	mux.HandleFunc("/query", srv.handleQuery)
	mux.HandleFunc("/status", srv.handleStatus)

	fmt.Printf("recall serving on http://%s\n", *addr)
	if err := http.ListenAndServe(*addr, mux); err != nil {
		fail("%v", err)
	}
}

type server struct {
	rag *rag.RAG
}

func newServer(r *rag.RAG) *server { return &server{rag: r} }

type queryRequest struct {
	Question string `json:"question"`
	K        int    `json:"k"`
}

type sourceJSON struct {
	Source string  `json:"source"`
	Page   int     `json:"page"`
	Score  float32 `json:"score"`
}

type queryResponse struct {
	Answer  string       `json:"answer"`
	Sources []sourceJSON `json:"sources"`
}

func (s *server) handleQuery(w http.ResponseWriter, req *http.Request) {
	if req.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}
	var qr queryRequest
	if err := json.NewDecoder(req.Body).Decode(&qr); err != nil {
		http.Error(w, "bad request: "+err.Error(), http.StatusBadRequest)
		return
	}
	if qr.K <= 0 {
		qr.K = 5
	}
	ans, err := s.rag.Ask(qr.Question, qr.K)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(toResponse(ans))
}

func (s *server) handleStatus(w http.ResponseWriter, req *http.Request) {
	st := s.rag.Store()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"files":  st.FileCount(),
		"chunks": st.ChunkCount(),
	})
}

func toResponse(ans rag.Answer) queryResponse {
	out := queryResponse{Answer: ans.Text}
	for _, s := range ans.Sources {
		out.Sources = append(out.Sources, sourceJSON{
			Source: s.Chunk.Source,
			Page:   s.Chunk.PageNum,
			Score:  s.Score,
		})
	}
	return out
}
