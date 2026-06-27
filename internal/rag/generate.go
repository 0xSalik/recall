package rag

import (
	"bufio"
	"bytes"
	"context"
	"fmt"
	"io"
	"os/exec"
	"strconv"
	"strings"
)

// GenOptions controls a single generation call.
type GenOptions struct {
	NPredict      int
	Temp          float64
	RepeatPenalty float64
}

// DefaultGenOptions matches the settings from the design: low temperature for
// faithful, grounded answers.
func DefaultGenOptions() GenOptions {
	return GenOptions{NPredict: 512, Temp: 0.1, RepeatPenalty: 1.1}
}

// Generator produces text from a prompt. Implementations may stream.
type Generator interface {
	// Generate returns the full completion for prompt.
	Generate(prompt string, opts GenOptions) (string, error)
	// GenerateStream emits incremental text on tokens as it is produced and
	// closes nothing (the caller owns the channel). It returns the full text.
	GenerateStream(ctx context.Context, prompt string, opts GenOptions, tokens chan<- string) (string, error)
}

// LlamaGenerator runs text generation via the `llama-cli` subprocess. Using a
// subprocess instead of CGO keeps the build portable; the cost is process
// startup latency per query, which is small next to generation time itself.
type LlamaGenerator struct {
	modelPath string
	llamaPath string
}

// NewLlamaGenerator resolves a generation binary. When llamaPath is empty it
// prefers llama-completion (the dedicated one-shot tool in current llama.cpp)
// and falls back to llama-cli for older single-binary builds. Note that in
// recent llama.cpp llama-cli is an interactive chat UI and is unsuitable for
// scripted one-shot generation, so llama-completion is strongly preferred.
func NewLlamaGenerator(modelPath, llamaPath string) (*LlamaGenerator, error) {
	if llamaPath == "" {
		for _, name := range []string{"llama-completion", "llama-cli"} {
			if p, err := exec.LookPath(name); err == nil {
				llamaPath = p
				break
			}
		}
		if llamaPath == "" {
			return nil, fmt.Errorf("rag: no generation binary found (need llama-completion or llama-cli in PATH)")
		}
	} else if _, err := exec.LookPath(llamaPath); err != nil {
		return nil, fmt.Errorf("rag: generation binary %q not found: %w", llamaPath, err)
	}
	return &LlamaGenerator{modelPath: modelPath, llamaPath: llamaPath}, nil
}

// args builds a non-interactive, subprocess-friendly invocation:
//   - -st / --single-turn   run one turn and exit (no interactive prompt)
//   - --simple-io           plain stdout suitable for capture (no spinner/banner)
//   - --no-display-prompt   emit only the completion, not the echoed prompt
//
// --log-disable is intentionally omitted; logs go to stderr (captured
// separately) and on some builds the flag also suppresses wanted output.
func (g *LlamaGenerator) args(prompt string, opts GenOptions) []string {
	return []string{
		"--model", g.modelPath,
		"--prompt", prompt,
		"--n-predict", itoa(opts.NPredict),
		"--temp", ftoa(opts.Temp),
		"--repeat-penalty", ftoa(opts.RepeatPenalty),
		"-st",
		"--simple-io",
		"--no-display-prompt",
	}
}

func (g *LlamaGenerator) Generate(prompt string, opts GenOptions) (string, error) {
	cmd := exec.Command(g.llamaPath, g.args(prompt, opts)...)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("rag: generation failed: %v: %s", err, strings.TrimSpace(stderr.String()))
	}
	return cleanAnswer(stdout.String()), nil
}

// GenerateStream runs llama-cli and forwards stdout to tokens as it arrives. The
// reader emits whatever bytes are currently buffered, which in practice gives
// token-granularity streaming since llama-cli flushes per token.
func (g *LlamaGenerator) GenerateStream(ctx context.Context, prompt string, opts GenOptions, tokens chan<- string) (string, error) {
	cmd := exec.CommandContext(ctx, g.llamaPath, g.args(prompt, opts)...)
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return "", err
	}
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Start(); err != nil {
		return "", err
	}

	var full strings.Builder
	reader := bufio.NewReader(stdout)
	buf := make([]byte, 256)
	for {
		n, rerr := reader.Read(buf)
		if n > 0 {
			piece := string(buf[:n])
			full.WriteString(piece)
			select {
			case tokens <- piece:
			case <-ctx.Done():
				_ = cmd.Process.Kill()
				return cleanAnswer(full.String()), ctx.Err()
			}
		}
		if rerr == io.EOF {
			break
		}
		if rerr != nil {
			return cleanAnswer(full.String()), rerr
		}
	}
	if err := cmd.Wait(); err != nil {
		return cleanAnswer(full.String()), fmt.Errorf("rag: generation failed: %v: %s", err, strings.TrimSpace(stderr.String()))
	}
	return cleanAnswer(full.String()), nil
}

// cleanAnswer strips common llama.cpp end-of-text artifacts and surrounding
// whitespace from a completion.
func cleanAnswer(s string) string {
	s = strings.TrimSpace(s)
	for _, marker := range []string{"[end of text]", "<|end|>", "<|endoftext|>", "</s>", "<|eot_id|>", "<|im_end|>"} {
		if i := strings.Index(s, marker); i >= 0 {
			s = s[:i]
		}
	}
	return strings.TrimSpace(s)
}

func itoa(i int) string { return strconv.Itoa(i) }

func ftoa(f float64) string { return strconv.FormatFloat(f, 'g', -1, 64) }
