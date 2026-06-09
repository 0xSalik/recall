package embed

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os/exec"
	"strconv"
	"strings"
)

// LlamaEmbedder embeds text by invoking the `llama-embedding` binary as a
// subprocess. Texts are passed on stdin (one per line) and vectors are read
// back from stdout. Output is parsed leniently to tolerate the format drift
// across llama.cpp versions (JSON array, OpenAI-style JSON, or whitespace
// floats).
type LlamaEmbedder struct {
	modelPath string
	llamaPath string // path to the llama-embedding binary
	dims      int
}

// NewLlamaEmbedder constructs a subprocess-backed embedder and probes the model
// once to discover its output dimensionality. It returns an error if the binary
// is missing or the probe fails.
func NewLlamaEmbedder(modelPath, llamaPath string) (*LlamaEmbedder, error) {
	if llamaPath == "" {
		llamaPath = "llama-embedding"
	}
	if _, err := exec.LookPath(llamaPath); err != nil {
		return nil, fmt.Errorf("embed: %q not found in PATH: %w", llamaPath, err)
	}
	e := &LlamaEmbedder{modelPath: modelPath, llamaPath: llamaPath}

	probe, err := e.run([]string{"recall embedding probe"})
	if err != nil {
		return nil, fmt.Errorf("embed: probe failed: %w", err)
	}
	if len(probe) == 0 || len(probe[0]) == 0 {
		return nil, fmt.Errorf("embed: probe returned no vector")
	}
	e.dims = len(probe[0])
	return e, nil
}

func (e *LlamaEmbedder) Dims() int { return e.dims }

// Embed normalizes and returns one vector per input, processing in batches.
func (e *LlamaEmbedder) Embed(texts []string) ([][]float32, error) {
	out := make([][]float32, 0, len(texts))
	for i := 0; i < len(texts); i += batchSize {
		end := i + batchSize
		if end > len(texts) {
			end = len(texts)
		}
		vecs, err := e.run(texts[i:end])
		if err != nil {
			return nil, err
		}
		if len(vecs) != end-i {
			return nil, fmt.Errorf("embed: expected %d vectors, got %d", end-i, len(vecs))
		}
		for _, v := range vecs {
			out = append(out, normalize(v))
		}
	}
	return out, nil
}

// run invokes the binary once for a batch of texts and returns parsed vectors.
func (e *LlamaEmbedder) run(texts []string) ([][]float32, error) {
	args := []string{
		"--model", e.modelPath,
		"--embd-normalize", "2",
		"--embd-output-format", "json",
		"--log-disable",
	}
	cmd := exec.Command(e.llamaPath, args...)

	// llama-embedding reads the prompt(s) from stdin; one per line.
	cmd.Stdin = strings.NewReader(strings.Join(sanitize(texts), "\n") + "\n")
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("embed: llama-embedding: %v: %s", err, strings.TrimSpace(stderr.String()))
	}
	return parseEmbeddings(stdout.Bytes())
}

// sanitize collapses newlines inside individual texts so the one-per-line stdin
// contract holds.
func sanitize(texts []string) []string {
	out := make([]string, len(texts))
	for i, t := range texts {
		t = strings.ReplaceAll(t, "\r\n", " ")
		t = strings.ReplaceAll(t, "\n", " ")
		out[i] = strings.TrimSpace(t)
	}
	return out
}

// parseEmbeddings handles the formats llama.cpp has emitted over time:
//   - OpenAI-style: {"object":"list","data":[{"embedding":[...]}, ...]}
//   - bare array of arrays: [[...],[...]]
//   - whitespace floats, one vector per line (legacy / non-JSON builds)
func parseEmbeddings(data []byte) ([][]float32, error) {
	trimmed := bytes.TrimSpace(data)
	if len(trimmed) == 0 {
		return nil, fmt.Errorf("embed: empty output")
	}

	switch trimmed[0] {
	case '{':
		var obj struct {
			Data []struct {
				Embedding []float32 `json:"embedding"`
			} `json:"data"`
		}
		if err := json.Unmarshal(trimmed, &obj); err == nil && len(obj.Data) > 0 {
			out := make([][]float32, len(obj.Data))
			for i, d := range obj.Data {
				out[i] = d.Embedding
			}
			return out, nil
		}
		// Single-object form: {"embedding":[...]}
		var single struct {
			Embedding []float32 `json:"embedding"`
		}
		if err := json.Unmarshal(trimmed, &single); err == nil && len(single.Embedding) > 0 {
			return [][]float32{single.Embedding}, nil
		}
		return nil, fmt.Errorf("embed: unrecognized JSON object output")
	case '[':
		// Either [[...],[...]] or a single flat [..].
		var nested [][]float32
		if err := json.Unmarshal(trimmed, &nested); err == nil && len(nested) > 0 {
			return nested, nil
		}
		var flat []float32
		if err := json.Unmarshal(trimmed, &flat); err == nil && len(flat) > 0 {
			return [][]float32{flat}, nil
		}
		return nil, fmt.Errorf("embed: unrecognized JSON array output")
	default:
		return parseWhitespaceFloats(trimmed)
	}
}

// parseWhitespaceFloats reads one vector per non-empty line of space-separated
// floats.
func parseWhitespaceFloats(data []byte) ([][]float32, error) {
	var out [][]float32
	for _, line := range strings.Split(string(data), "\n") {
		fields := strings.Fields(line)
		if len(fields) == 0 {
			continue
		}
		vec := make([]float32, 0, len(fields))
		for _, f := range fields {
			x, err := strconv.ParseFloat(f, 32)
			if err != nil {
				// A stray non-numeric line (e.g. a log leak): skip it.
				vec = nil
				break
			}
			vec = append(vec, float32(x))
		}
		if len(vec) > 0 {
			out = append(out, vec)
		}
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("embed: no vectors parsed from output")
	}
	return out, nil
}
