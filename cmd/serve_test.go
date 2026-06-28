package cmd

import (
	"context"
	"encoding/json"
	"hash/fnv"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/0xSalik/recall/internal/rag"
	"github.com/0xSalik/recall/internal/store"
)

type fakeEmbedder struct{ dims int }

func (f fakeEmbedder) Dims() int { return f.dims }
func (f fakeEmbedder) Embed(texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i, t := range texts {
		v := make([]float32, f.dims)
		for _, tok := range strings.Fields(strings.ToLower(t)) {
			h := fnv.New32a()
			h.Write([]byte(tok))
			v[h.Sum32()%uint32(f.dims)] += 1
		}
		var sum float64
		for _, x := range v {
			sum += float64(x) * float64(x)
		}
		if sum > 0 {
			inv := float32(1 / math.Sqrt(sum))
			for j := range v {
				v[j] *= inv
			}
		}
		out[i] = v
	}
	return out, nil
}

type fakeGenerator struct{}

func (fakeGenerator) Generate(prompt string, opts rag.GenOptions) (string, error) {
	return "the answer", nil
}
func (fakeGenerator) GenerateStream(ctx context.Context, prompt string, opts rag.GenOptions, tokens chan<- string) (string, error) {
	for _, w := range []string{"the", " answer"} {
		tokens <- w
	}
	return "the answer", nil
}

func newTestServer(t *testing.T) *server {
	t.Helper()
	st, err := store.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	r := rag.NewWithComponents(st, fakeEmbedder{dims: 128}, fakeGenerator{})

	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "doc.txt"),
		[]byte("vectors are embedded and indexed for retrieval and ranking"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := r.Index([]string{dir}); err != nil {
		t.Fatal(err)
	}
	return newServer(r)
}

func TestServeJSONQuery(t *testing.T) {
	srv := newTestServer(t)
	body, _ := json.Marshal(queryRequest{Question: "how are vectors retrieved", K: 3})
	req := httptest.NewRequest(http.MethodPost, "/query", strings.NewReader(string(body)))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	srv.handleQuery(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	var resp queryResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if resp.Answer != "the answer" {
		t.Fatalf("answer = %q", resp.Answer)
	}
	if len(resp.Sources) == 0 {
		t.Fatal("expected sources")
	}
}

func TestServeSSEQuery(t *testing.T) {
	srv := newTestServer(t)
	body, _ := json.Marshal(queryRequest{Question: "how are vectors retrieved", K: 3})
	req := httptest.NewRequest(http.MethodPost, "/query", strings.NewReader(string(body)))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")
	rec := httptest.NewRecorder()

	srv.handleQuery(rec, req)

	if ct := rec.Header().Get("Content-Type"); ct != "text/event-stream" {
		t.Fatalf("content-type = %q", ct)
	}
	out := rec.Body.String()

	// Parse the SSE events back out.
	var tokenText strings.Builder
	sawDone := false
	for _, block := range strings.Split(out, "\n\n") {
		for _, line := range strings.Split(block, "\n") {
			if !strings.HasPrefix(line, "data: ") {
				continue
			}
			payload := strings.TrimPrefix(line, "data: ")
			var ev map[string]any
			if err := json.Unmarshal([]byte(payload), &ev); err != nil {
				t.Fatalf("bad event %q: %v", payload, err)
			}
			if tok, ok := ev["token"].(string); ok {
				tokenText.WriteString(tok)
			}
			if d, _ := ev["done"].(bool); d {
				sawDone = true
				if _, ok := ev["sources"]; !ok {
					t.Fatal("terminal event missing sources")
				}
			}
		}
	}
	if tokenText.String() != "the answer" {
		t.Fatalf("streamed tokens = %q, want %q", tokenText.String(), "the answer")
	}
	if !sawDone {
		t.Fatal("never received terminal done event")
	}
}

func TestServeStatus(t *testing.T) {
	srv := newTestServer(t)
	req := httptest.NewRequest(http.MethodGet, "/status", nil)
	rec := httptest.NewRecorder()
	srv.handleStatus(rec, req)
	var s map[string]int
	if err := json.Unmarshal(rec.Body.Bytes(), &s); err != nil {
		t.Fatal(err)
	}
	if s["chunks"] == 0 {
		t.Fatal("expected chunks > 0")
	}
}

func TestServeFilesAndRemove(t *testing.T) {
	srv := newTestServer(t)

	// /files lists the one indexed doc.
	req := httptest.NewRequest(http.MethodGet, "/files", nil)
	rec := httptest.NewRecorder()
	srv.handleFiles(rec, req)
	var fl struct {
		Files []store.FileInfo `json:"files"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &fl); err != nil {
		t.Fatal(err)
	}
	if len(fl.Files) != 1 {
		t.Fatalf("expected 1 file, got %d", len(fl.Files))
	}

	// /remove drops it.
	body, _ := json.Marshal(removeRequest{Path: fl.Files[0].Path})
	rreq := httptest.NewRequest(http.MethodPost, "/remove", strings.NewReader(string(body)))
	rrec := httptest.NewRecorder()
	srv.handleRemove(rrec, rreq)
	if rrec.Code != http.StatusOK {
		t.Fatalf("remove status = %d: %s", rrec.Code, rrec.Body.String())
	}
	if srv.rag.Store().FileCount() != 0 {
		t.Fatalf("file not removed, count = %d", srv.rag.Store().FileCount())
	}
}

func TestServeClear(t *testing.T) {
	srv := newTestServer(t)
	req := httptest.NewRequest(http.MethodPost, "/clear", nil)
	rec := httptest.NewRecorder()
	srv.handleClear(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("clear status = %d", rec.Code)
	}
	if srv.rag.Store().ChunkCount() != 0 {
		t.Fatal("store not cleared")
	}
}

func TestServeRefreshPrunesDeleted(t *testing.T) {
	st, err := store.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	r := rag.NewWithComponents(st, fakeEmbedder{dims: 128}, fakeGenerator{})
	dir := t.TempDir()
	p := filepath.Join(dir, "doc.txt")
	if err := os.WriteFile(p, []byte("vectors are embedded and indexed for retrieval and ranking"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := r.Index([]string{dir}); err != nil {
		t.Fatal(err)
	}
	srv := newServer(r)

	if err := os.Remove(p); err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest(http.MethodPost, "/refresh", nil)
	rec := httptest.NewRecorder()
	srv.handleRefresh(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("refresh status = %d: %s", rec.Code, rec.Body.String())
	}
	if srv.rag.Store().FileCount() != 0 {
		t.Fatalf("deleted file not pruned, count = %d", srv.rag.Store().FileCount())
	}
}

func TestServeIndexHTML(t *testing.T) {
	srv := newTestServer(t)
	req := httptest.NewRequest(http.MethodGet, "/", nil)
	rec := httptest.NewRecorder()
	srv.handleIndex(rec, req)
	if !strings.Contains(rec.Body.String(), "<title>recall</title>") {
		t.Fatal("index html not served")
	}
}
