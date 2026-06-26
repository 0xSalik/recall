// Command bench is a standalone benchmark harness for recall's search layer.
//
// It measures what can be measured without models present: index search
// latency percentiles, HNSW recall@5 against the exact flat index, and memory
// footprint of the index. Embedding and generation throughput require the
// llama.cpp models and are reported as such here; use `recall query` with a
// model installed to observe end-to-end latency.
package main

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"sort"
	"strconv"
	"time"

	"github.com/0xSalik/recall/internal/index"
)

func normalize(v []float32) []float32 {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	if sum == 0 {
		return v
	}
	inv := float32(1 / math.Sqrt(sum))
	for i := range v {
		v[i] *= inv
	}
	return v
}

func unitVec(r *rand.Rand, dims int) []float32 {
	v := make([]float32, dims)
	for i := range v {
		v[i] = float32(r.NormFloat64())
	}
	return normalize(v)
}

// clusteredVecs models a realistic embedding distribution: points drawn around
// a set of random centroids rather than uniformly over the sphere. Uniform
// random vectors have near-tied neighbors and are a pathological case for any
// approximate index; real embeddings cluster, which is what makes ANN useful.
func clusteredVecs(r *rand.Rand, n, dims, clusters int, spread float64) [][]float32 {
	centroids := make([][]float32, clusters)
	for c := range centroids {
		centroids[c] = unitVec(r, dims)
	}
	out := make([][]float32, n)
	for i := range out {
		c := centroids[r.Intn(clusters)]
		v := make([]float32, dims)
		for j := range v {
			v[j] = c[j] + float32(r.NormFloat64()*spread)
		}
		out[i] = normalize(v)
	}
	return out
}

func percentile(sorted []time.Duration, p float64) time.Duration {
	if len(sorted) == 0 {
		return 0
	}
	idx := int(math.Ceil(p/100*float64(len(sorted)))) - 1
	if idx < 0 {
		idx = 0
	}
	if idx >= len(sorted) {
		idx = len(sorted) - 1
	}
	return sorted[idx]
}

type latencyStats struct {
	p50, p95, p99 time.Duration
}

func measureLatency(idx index.Index, queries [][]float32, k int) latencyStats {
	durs := make([]time.Duration, len(queries))
	for i, q := range queries {
		start := time.Now()
		idx.Search(q, k)
		durs[i] = time.Since(start)
	}
	sort.Slice(durs, func(i, j int) bool { return durs[i] < durs[j] })
	return latencyStats{
		p50: percentile(durs, 50),
		p95: percentile(durs, 95),
		p99: percentile(durs, 99),
	}
}

func buildBoth(vecs [][]float32, dims int) (*index.FlatIndex, *index.HNSW) {
	flat := index.NewFlatIndex(dims)
	h := index.NewHNSW(dims)
	for i, v := range vecs {
		id := "id-" + strconv.Itoa(i)
		flat.Add(id, v, i)
		h.Add(id, v, i)
	}
	return flat, h
}

// recallAt5 computes the fraction of HNSW's top-5 that appear in the flat
// index's ground-truth top-5, averaged over the query set.
func recallAt5(flat *index.FlatIndex, h *index.HNSW, queries [][]float32) float64 {
	var total float64
	for _, q := range queries {
		fres, _ := flat.Search(q, 5)
		hres, _ := h.Search(q, 5)
		truth := map[string]bool{}
		for _, r := range fres {
			truth[r.ID] = true
		}
		hit := 0
		for _, r := range hres {
			if truth[r.ID] {
				hit++
			}
		}
		if len(fres) > 0 {
			total += float64(hit) / float64(len(fres))
		}
	}
	return total / float64(len(queries))
}

func main() {
	dims := 128
	sizes := []int{1000, 10000}
	queryCount := 500

	fmt.Println("Benchmark Results")
	fmt.Println("-----------------")
	fmt.Printf("Vector dims: %d, queries per measurement: %d\n\n", dims, queryCount)

	const clusters = 64
	const spread = 0.06

	qr := rand.New(rand.NewSource(99))
	queries := clusteredVecs(qr, queryCount, dims, clusters, spread)

	fmt.Println("Index search latency (k=5), clustered vectors:")
	fmt.Printf("  %-8s %-32s %-32s %-10s\n", "n", "flat (p50/p95/p99)", "hnsw (p50/p95/p99)", "recall@5")
	for _, n := range sizes {
		vr := rand.New(rand.NewSource(int64(n)))
		vecs := clusteredVecs(vr, n, dims, clusters, spread)

		var before runtime.MemStats
		runtime.GC()
		runtime.ReadMemStats(&before)

		flat, h := buildBoth(vecs, dims)

		var after runtime.MemStats
		runtime.ReadMemStats(&after)
		indexMem := after.HeapAlloc - before.HeapAlloc

		fs := measureLatency(flat, queries, 5)
		hs := measureLatency(h, queries, 5)
		recall := recallAt5(flat, h, queries)

		fmt.Printf("  %-8d %-32s %-32s %.1f%%\n",
			n,
			fmt.Sprintf("%s/%s/%s", fs.p50, fs.p95, fs.p99),
			fmt.Sprintf("%s/%s/%s", hs.p50, hs.p95, hs.p99),
			recall*100,
		)
		fmt.Printf("           index memory (both): %s\n", humanBytes(int64(indexMem)))
	}

	fmt.Println()
	fmt.Println("Embedding throughput / generation latency:")
	fmt.Println("  requires llama.cpp + GGUF models; run `recall query --stream` to observe live.")
}

func humanBytes(n int64) string {
	const unit = 1024
	if n < unit {
		return fmt.Sprintf("%d B", n)
	}
	div, exp := int64(unit), 0
	for x := n / unit; x >= unit; x /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %cB", float64(n)/float64(div), "KMGTPE"[exp])
}
