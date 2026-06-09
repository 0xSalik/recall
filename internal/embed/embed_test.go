package embed

import (
	"hash/fnv"
	"math"
	"strings"
	"testing"
)

// fakeEmbedder is a deterministic bag-of-words embedder used to exercise the
// Embedder contract (length, normalization, relative similarity) without a real
// model. Shared tokens between texts push their vectors together, so similar
// texts get higher cosine similarity — enough to validate the pipeline.
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
		out[i] = normalize(v)
	}
	return out, nil
}

func cosine(a, b []float32) float64 {
	var dot float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
	}
	return dot
}

func TestFakeEmbedderContract(t *testing.T) {
	e := fakeEmbedder{dims: 256}
	vecs, err := e.Embed([]string{"hello world"})
	if err != nil {
		t.Fatal(err)
	}
	if len(vecs) != 1 || len(vecs[0]) != 256 {
		t.Fatalf("unexpected shape: %d x %d", len(vecs), len(vecs[0]))
	}
	if n := l2norm(vecs[0]); math.Abs(n-1.0) > 1e-6 {
		t.Fatalf("vector not unit length: %f", n)
	}
}

func TestSimilarityOrdering(t *testing.T) {
	e := fakeEmbedder{dims: 512}
	vecs, err := e.Embed([]string{
		"the cat sat on the mat",
		"a cat sat on a mat today",
		"quantum chromodynamics and gluon fields",
	})
	if err != nil {
		t.Fatal(err)
	}
	simSimilar := cosine(vecs[0], vecs[1])
	simDifferent := cosine(vecs[0], vecs[2])
	if simSimilar <= simDifferent {
		t.Fatalf("expected similar texts to score higher: similar=%f different=%f", simSimilar, simDifferent)
	}
}

func TestNormalizeUnitLength(t *testing.T) {
	v := []float32{3, 4} // norm 5
	normalize(v)
	if n := l2norm(v); math.Abs(n-1.0) > 1e-6 {
		t.Fatalf("norm = %f, want 1.0", n)
	}
	// self dot product of a unit vector ~= 1.0
	if d := cosine(v, v); math.Abs(d-1.0) > 1e-6 {
		t.Fatalf("self dot = %f, want 1.0", d)
	}
}

func TestNormalizeZeroVector(t *testing.T) {
	v := []float32{0, 0, 0}
	normalize(v)
	for _, x := range v {
		if x != 0 {
			t.Fatal("zero vector should remain zero")
		}
	}
}

func TestParseEmbeddingsFormats(t *testing.T) {
	cases := map[string]string{
		"openai":       `{"object":"list","data":[{"embedding":[0.1,0.2,0.3]},{"embedding":[0.4,0.5,0.6]}]}`,
		"nested_array": `[[0.1,0.2,0.3],[0.4,0.5,0.6]]`,
		"single_obj":   `{"embedding":[0.1,0.2,0.3]}`,
		"flat_array":   `[0.1,0.2,0.3]`,
		"whitespace":   "0.1 0.2 0.3\n0.4 0.5 0.6\n",
	}
	for name, out := range cases {
		t.Run(name, func(t *testing.T) {
			vecs, err := parseEmbeddings([]byte(out))
			if err != nil {
				t.Fatalf("parse failed: %v", err)
			}
			if len(vecs) == 0 || len(vecs[0]) != 3 {
				t.Fatalf("unexpected parse result: %v", vecs)
			}
		})
	}
}

func TestParseServerResponseFormats(t *testing.T) {
	cases := []string{
		`[{"embedding":[0.1,0.2,0.3]}]`,
		`{"embedding":[0.1,0.2,0.3]}`,
		`{"data":[{"embedding":[0.1,0.2,0.3]}]}`,
		`[{"embedding":[[0.1,0.2,0.3]]}]`,
	}
	for _, c := range cases {
		vecs, err := parseServerResponse([]byte(c), 1)
		if err != nil {
			t.Fatalf("parse failed for %q: %v", c, err)
		}
		if len(vecs) != 1 || len(vecs[0]) != 3 {
			t.Fatalf("unexpected result for %q: %v", c, vecs)
		}
	}
}

func TestDetectNoBackend(t *testing.T) {
	// A binary name that won't exist and no server URL.
	_, err := Detect("models/nonexistent.gguf", "llama-embedding-does-not-exist-xyz", "")
	if err == nil {
		t.Fatal("expected error when no backend is available")
	}
}
