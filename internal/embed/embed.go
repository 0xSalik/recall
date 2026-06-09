// Package embed turns text into normalized embedding vectors.
//
// The default backend shells out to llama.cpp. Two transports are supported and
// auto-detected: the `llama-embedding` CLI (one process per batch, stdin in /
// vectors out) and a running `llama-server --embedding` HTTP endpoint. Using a
// subprocess instead of CGO keeps the build dependency-free and portable at the
// cost of a little per-call latency.
//
// All vectors returned by this package are L2-normalized, so cosine similarity
// downstream reduces to a dot product.
package embed

import (
	"math"
)

// Embedder produces fixed-dimension embeddings for a batch of texts.
type Embedder interface {
	// Embed returns one normalized vector per input text, in order.
	Embed(texts []string) ([][]float32, error)
	// Dims is the dimensionality of the vectors this embedder produces.
	Dims() int
}

// batchSize bounds how many texts we hand to a single backend invocation so a
// large ingest run doesn't OOM the model process.
const batchSize = 32

// normalize scales v to unit L2 length in place and returns it. A zero vector
// is left unchanged (it has no direction to preserve).
func normalize(v []float32) []float32 {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	if sum == 0 {
		return v
	}
	inv := float32(1.0 / math.Sqrt(sum))
	for i := range v {
		v[i] *= inv
	}
	return v
}

// l2norm returns the L2 norm of v (used by tests and sanity checks).
func l2norm(v []float32) float64 {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	return math.Sqrt(sum)
}
