package cmd

import (
	"encoding/json"
	"flag"
	"fmt"
	"net/http"
	"strings"

	"github.com/0xSalik/recall/internal/rag"
	"github.com/0xSalik/recall/internal/store"
	"github.com/0xSalik/recall/web"
)

// Serve implements `recall serve`: a local HTTP server that serves the embedded
// browser UI and exposes a JSON + SSE API over the RAG pipeline.
//
// POST /query negotiates on the Accept header:
//   - Accept: text/event-stream  -> Server-Sent Events, one event per token,
//     terminated by a {"done":true, "sources":[...]} event.
//   - otherwise                  -> a single JSON {answer, sources} response,
//     which is what the CLI and simple clients use.
func Serve(args []string) {
	fs := flag.NewFlagSet("serve", flag.ExitOnError)
	addr := fs.String("addr", "localhost:8080", "listen address")
	store := fs.String("store", defaultStoreDir(), "path to store directory")
	embedModel := fs.String("embed", "", modelFlagHelp)
	genModel := fs.String("gen", "", modelFlagHelp)
	llama := fs.String("llama", "", "path to generation binary (default: search PATH)")
	bin := fs.String("bin", "", "directory containing llama.cpp binaries (prepended to PATH)")
	fs.Parse(args)
	resolveEngine(*bin)

	r, err := rag.New(*store, resolveEmbedModel(*embedModel), resolveGenModel(*genModel), *llama)
	if err != nil {
		fail("%v", err)
	}

	srv := newServer(r)
	mux := http.NewServeMux()
	mux.HandleFunc("/", srv.handleIndex)
	mux.HandleFunc("/query", srv.handleQuery)
	mux.HandleFunc("/status", srv.handleStatus)
	mux.HandleFunc("/files", srv.handleFiles)
	mux.HandleFunc("/remove", srv.handleRemove)
	mux.HandleFunc("/refresh", srv.handleRefresh)
	mux.HandleFunc("/clear", srv.handleClear)

	fmt.Printf("recall serving on http://%s\n", *addr)
	if err := http.ListenAndServe(*addr, mux); err != nil {
		fail("%v", err)
	}
}

type server struct {
	rag *rag.RAG
}

func newServer(r *rag.RAG) *server { return &server{rag: r} }

func (s *server) handleIndex(w http.ResponseWriter, req *http.Request) {
	if req.URL.Path != "/" {
		http.NotFound(w, req)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Write(web.IndexHTML)
}

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
	if req.Header.Get("Accept") == "text/event-stream" {
		s.streamQuery(w, req, qr)
		return
	}
	s.jsonQuery(w, qr)
}

// jsonQuery runs the full pipeline and returns a single JSON response. This is
// the non-streaming path used by the CLI.
func (s *server) jsonQuery(w http.ResponseWriter, qr queryRequest) {
	ans, err := s.rag.Ask(qr.Question, qr.K)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(toResponse(ans))
}

// sseTokenEvent is a per-token event; sseDoneEvent is the terminal event.
type sseTokenEvent struct {
	Token string `json:"token"`
	Done  bool   `json:"done"`
}

type sseDoneEvent struct {
	Sources []sourceJSON `json:"sources"`
	Done    bool         `json:"done"`
}

// streamQuery streams generation token-by-token as Server-Sent Events. Tokens
// arrive on a channel from rag.AskStream (which pipes llama-cli stdout); each is
// written as its own event and flushed immediately so the browser renders it as
// it lands.
func (s *server) streamQuery(w http.ResponseWriter, req *http.Request, qr queryRequest) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming unsupported", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	enc := json.NewEncoder(w)
	writeEvent := func(v any) {
		fmt.Fprint(w, "data: ")
		enc.Encode(v) // Encode appends a newline
		fmt.Fprint(w, "\n")
		flusher.Flush()
	}

	tokens := make(chan string, 32)
	drained := make(chan struct{})
	go func() {
		for tok := range tokens {
			writeEvent(sseTokenEvent{Token: tok, Done: false})
		}
		close(drained)
	}()

	ans, err := s.rag.AskStream(req.Context(), qr.Question, qr.K, tokens)
	close(tokens)
	<-drained // ensure no concurrent writes to w before the terminal event

	if err != nil {
		writeEvent(map[string]any{"error": err.Error(), "done": true})
		return
	}
	writeEvent(sseDoneEvent{Sources: toResponse(ans).Sources, Done: true})
}

func (s *server) handleStatus(w http.ResponseWriter, req *http.Request) {
	st := s.rag.Store()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"files":  st.FileCount(),
		"chunks": st.ChunkCount(),
	})
}

// handleFiles returns the indexed files with chunk counts.
func (s *server) handleFiles(w http.ResponseWriter, req *http.Request) {
	files := s.rag.ListFiles()
	if files == nil {
		files = []store.FileInfo{}
	}
	writeJSON(w, map[string]any{"files": files})
}

type removeRequest struct {
	Path string `json:"path"`
}

// handleRemove drops a file or folder from the index.
func (s *server) handleRemove(w http.ResponseWriter, req *http.Request) {
	if req.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}
	var rr removeRequest
	if err := json.NewDecoder(req.Body).Decode(&rr); err != nil || strings.TrimSpace(rr.Path) == "" {
		http.Error(w, "bad request: path required", http.StatusBadRequest)
		return
	}
	n, files, err := s.rag.Remove(rr.Path)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	writeJSON(w, map[string]any{"removedFiles": len(files), "removedChunks": n})
}

type refreshRequest struct {
	Paths []string `json:"paths"`
}

// handleRefresh prunes deleted files, reindexes changed ones, and indexes new
// files under any provided paths.
func (s *server) handleRefresh(w http.ResponseWriter, req *http.Request) {
	if req.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}
	var rr refreshRequest
	// Body is optional; ignore decode errors on empty bodies.
	_ = json.NewDecoder(req.Body).Decode(&rr)
	res, err := s.rag.Refresh(rr.Paths)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	indexed := 0
	for _, ir := range res.Reindexed {
		if !ir.Skipped {
			indexed++
		}
	}
	writeJSON(w, map[string]any{"pruned": len(res.Deleted), "indexed": indexed})
}

// handleClear empties the index.
func (s *server) handleClear(w http.ResponseWriter, req *http.Request) {
	if req.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}
	if err := s.rag.Clear(); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	writeJSON(w, map[string]any{"ok": true})
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(v)
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
