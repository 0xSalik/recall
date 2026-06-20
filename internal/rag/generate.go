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

// NewLlamaGenerator verifies the llama-cli binary is available.
func NewLlamaGenerator(modelPath, llamaPath string) (*LlamaGenerator, error) {
	if llamaPath == "" {
		llamaPath = "llama-cli"
	}
	if _, err := exec.LookPath(llamaPath); err != nil {
		return nil, fmt.Errorf("rag: generation binary %q not found: %w", llamaPath, err)
	}
	return &LlamaGenerator{modelPath: modelPath, llamaPath: llamaPath}, nil
}

func (g *LlamaGenerator) args(prompt string, opts GenOptions) []string {
	return []string{
		"--model", g.modelPath,
		"--prompt", prompt,
		"--n-predict", itoa(opts.NPredict),
		"--temp", ftoa(opts.Temp),
		"--repeat-penalty", ftoa(opts.RepeatPenalty),
		"--log-disable",
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
