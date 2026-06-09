package embed

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os/exec"
	"time"
)

// ServerEmbedder embeds text against a running `llama-server --embedding`
// instance over HTTP. This is the fallback when the llama-embedding CLI isn't
// available but a server is.
type ServerEmbedder struct {
	baseURL string
	client  *http.Client
	dims    int
}

// NewServerEmbedder connects to a llama-server embedding endpoint (e.g.
// "http://localhost:8081") and probes it for dimensionality.
func NewServerEmbedder(baseURL string) (*ServerEmbedder, error) {
	e := &ServerEmbedder{
		baseURL: baseURL,
		client:  &http.Client{Timeout: 60 * time.Second},
	}
	probe, err := e.embedBatch([]string{"recall embedding probe"})
	if err != nil {
		return nil, fmt.Errorf("embed: server probe failed: %w", err)
	}
	if len(probe) == 0 || len(probe[0]) == 0 {
		return nil, fmt.Errorf("embed: server probe returned no vector")
	}
	e.dims = len(probe[0])
	return e, nil
}

func (e *ServerEmbedder) Dims() int { return e.dims }

func (e *ServerEmbedder) Embed(texts []string) ([][]float32, error) {
	out := make([][]float32, 0, len(texts))
	for i := 0; i < len(texts); i += batchSize {
		end := i + batchSize
		if end > len(texts) {
			end = len(texts)
		}
		vecs, err := e.embedBatch(texts[i:end])
		if err != nil {
			return nil, err
		}
		for _, v := range vecs {
			out = append(out, normalize(v))
		}
	}
	return out, nil
}

// embedBatch posts a batch to the /embedding endpoint. llama-server accepts
// {"content": [...]} and returns [{"embedding":[...]}] (or a single object).
func (e *ServerEmbedder) embedBatch(texts []string) ([][]float32, error) {
	body, _ := json.Marshal(map[string]any{"content": texts})
	req, err := http.NewRequest(http.MethodPost, e.baseURL+"/embedding", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := e.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	data, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("embed: server returned %d: %s", resp.StatusCode, string(data))
	}
	return parseServerResponse(data, len(texts))
}

// parseServerResponse decodes the several response shapes llama-server has used
// for the /embedding endpoint.
func parseServerResponse(data []byte, want int) ([][]float32, error) {
	// Shape A: [{"embedding":[...]}, ...]
	var arr []struct {
		Embedding json.RawMessage `json:"embedding"`
	}
	if err := json.Unmarshal(data, &arr); err == nil && len(arr) > 0 {
		out := make([][]float32, 0, len(arr))
		for _, item := range arr {
			v, perr := decodeEmbeddingField(item.Embedding)
			if perr != nil {
				return nil, perr
			}
			out = append(out, v)
		}
		return out, nil
	}
	// Shape B: {"embedding":[...]} (single)
	var single struct {
		Embedding json.RawMessage `json:"embedding"`
	}
	if err := json.Unmarshal(data, &single); err == nil && len(single.Embedding) > 0 {
		v, perr := decodeEmbeddingField(single.Embedding)
		if perr != nil {
			return nil, perr
		}
		return [][]float32{v}, nil
	}
	// Shape C: OpenAI-style {"data":[{"embedding":[...]}]}
	var openai struct {
		Data []struct {
			Embedding []float32 `json:"embedding"`
		} `json:"data"`
	}
	if err := json.Unmarshal(data, &openai); err == nil && len(openai.Data) > 0 {
		out := make([][]float32, len(openai.Data))
		for i, d := range openai.Data {
			out[i] = d.Embedding
		}
		return out, nil
	}
	return nil, fmt.Errorf("embed: unrecognized server response: %s", truncate(string(data), 120))
}

// decodeEmbeddingField handles the "embedding" field being either a flat array
// [..] or a nested [[..]] (some builds wrap per-token embeddings).
func decodeEmbeddingField(raw json.RawMessage) ([]float32, error) {
	var flat []float32
	if err := json.Unmarshal(raw, &flat); err == nil && len(flat) > 0 {
		return flat, nil
	}
	var nested [][]float32
	if err := json.Unmarshal(raw, &nested); err == nil && len(nested) > 0 {
		return nested[0], nil
	}
	return nil, fmt.Errorf("embed: could not decode embedding field")
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}

// Detect picks an available backend: it prefers the llama-embedding CLI, then
// falls back to a llama-server endpoint at serverURL. modelPath is required for
// the CLI backend; serverURL may be empty to skip the server probe.
func Detect(modelPath, llamaPath, serverURL string) (Embedder, error) {
	if llamaPath == "" {
		llamaPath = "llama-embedding"
	}
	if _, err := exec.LookPath(llamaPath); err == nil {
		if e, cerr := NewLlamaEmbedder(modelPath, llamaPath); cerr == nil {
			return e, nil
		}
	}
	if serverURL != "" {
		if e, serr := NewServerEmbedder(serverURL); serr == nil {
			return e, nil
		}
	}
	return nil, fmt.Errorf("embed: no embedding backend available (need %q in PATH or a llama-server at %q)", llamaPath, serverURL)
}
