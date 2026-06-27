package embed

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
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

// promptSeparator is a sentinel used to delimit prompts in the batch file. It
// must not occur in normal text. We can't use a newline: current llama.cpp
// splits each file line *again* on its --embd-separator (whose default matches
// the literal two-character "\n"), so any chunk containing a literal "\n" (very
// common in source code, e.g. fmt strings) would be over-split, yielding more
// vectors than inputs. Joining on a unique sentinel and writing a single line
// avoids both the per-line and the literal-"\n" splitting.
const promptSeparator = "<#recall-sep#>"

// run invokes the binary once for a batch of texts and returns parsed vectors.
// The batch is written to a temp file (passed with -f) as a single line with
// prompts delimited by promptSeparator, which is also handed to the binary as
// --embd-separator so we get exactly one vector per input.
func (e *LlamaEmbedder) run(texts []string) ([][]float32, error) {
	f, err := os.CreateTemp("", "recall-embed-*.txt")
	if err != nil {
		return nil, fmt.Errorf("embed: temp file: %w", err)
	}
	defer os.Remove(f.Name())
	if _, err := f.WriteString(strings.Join(sanitize(texts), promptSeparator) + "\n"); err != nil {
		f.Close()
		return nil, fmt.Errorf("embed: writing prompts: %w", err)
	}
	f.Close()

	// NB: do not pass --log-disable here. In current llama.cpp the embedding
	// vectors are emitted through the logging system, so --log-disable silently
	// suppresses the output we need to parse. Diagnostic logs go to stderr,
	// which we capture separately, so stdout stays clean JSON.
	args := []string{
		"--model", e.modelPath,
		"-f", f.Name(),
		"--embd-normalize", "2",
		"--embd-output-format", "json",
		"--embd-separator", promptSeparator,
	}
	cmd := exec.Command(e.llamaPath, args...)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("embed: llama-embedding: %v: %s", err, strings.TrimSpace(stderr.String()))
	}
	return parseEmbeddings(stdout.Bytes())
}

// sanitize collapses newline variants to spaces and removes any occurrence of
// the prompt separator so each text stays a single, indivisible prompt.
func sanitize(texts []string) []string {
	replacer := strings.NewReplacer(
		"\r\n", " ",
		"\n", " ",
		"\r", " ",
		promptSeparator, " ",
	)
	out := make([]string, len(texts))
	for i, t := range texts {
		out[i] = strings.TrimSpace(replacer.Replace(t))
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
