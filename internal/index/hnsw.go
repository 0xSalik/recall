package index

import (
	"container/heap"
	"encoding/gob"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
)

// node is one element of the HNSW graph. Neighbor lists are stored as integer
// indices into HNSW.nodes (not pointers) so the whole structure gob-encodes
// without custom marshaling. Layers[0] is the dense base layer.
type node struct {
	ID       string
	Vec      []float32
	ChunkIdx int
	Layers   [][]int // Layers[l] = neighbor node indices at layer l
}

// HNSW is a hierarchical navigable small-world graph for approximate nearest
// neighbor search. Built from scratch following Malkov & Yashunin (2018).
type HNSW struct {
	nodes      []*node
	entryPoint int     // index into nodes; -1 when empty
	maxLayer   int     // highest layer currently populated
	M          int     // target connections per layer
	Mmax0      int     // max connections at layer 0 (denser by design)
	efConst    int     // candidate list size during construction
	efSearch   int     // candidate list size during search
	ml         float64 // level-generation normalization factor 1/ln(M)
	dims       int
	rng        *rand.Rand
}

// NewHNSW creates an empty graph with sensible defaults (M=16, Mmax0=32,
// efConstruction=200). dims may be 0 and is inferred from the first vector.
func NewHNSW(dims int) *HNSW {
	return newHNSWSeeded(dims, 1)
}

func newHNSWSeeded(dims int, seed int64) *HNSW {
	m := 16
	return &HNSW{
		entryPoint: -1,
		maxLayer:   0,
		M:          m,
		Mmax0:      2 * m,
		efConst:    200,
		efSearch:   100,
		ml:         1.0 / math.Log(float64(m)),
		dims:       dims,
		rng:        rand.New(rand.NewSource(seed)),
	}
}

func (h *HNSW) Len() int { return len(h.nodes) }

// visitedSet is an O(1)-reset visited marker: a node is "visited" when its stamp
// equals the current epoch. Reusing one set across a search avoids the
// allocation churn of a fresh map per layer. Not safe for concurrent use; each
// search/insert owns its own set.
type visitedSet struct {
	stamp []uint32
	epoch uint32
}

func (v *visitedSet) reset(n int) {
	if cap(v.stamp) < n {
		v.stamp = make([]uint32, n)
	} else {
		v.stamp = v.stamp[:n]
	}
	v.epoch++
	if v.epoch == 0 { // wrapped around; clear and restart
		for i := range v.stamp {
			v.stamp[i] = 0
		}
		v.epoch = 1
	}
}

func (v *visitedSet) test(i int) bool { return v.stamp[i] == v.epoch }
func (v *visitedSet) set(i int)        { v.stamp[i] = v.epoch }

// distance is cosine distance for unit vectors: 1 - dot. Smaller is closer.
func (h *HNSW) distance(a, b []float32) float32 {
	return 1 - dot(a, b)
}

// randomLevel draws a level from the exponentially-decaying distribution that
// gives HNSW its layered structure.
func (h *HNSW) randomLevel() int {
	r := h.rng.Float64()
	if r <= 0 {
		r = math.SmallestNonzeroFloat64
	}
	return int(math.Floor(-math.Log(r) * h.ml))
}

func (h *HNSW) Add(id string, vec []float32, chunkIdx int) error {
	if h.dims == 0 {
		h.dims = len(vec)
	}
	if len(vec) != h.dims {
		return fmt.Errorf("hnsw: vector dim %d != index dim %d", len(vec), h.dims)
	}
	cp := make([]float32, len(vec))
	copy(cp, vec)

	level := h.randomLevel()
	newIdx := len(h.nodes)
	n := &node{
		ID:       id,
		Vec:      cp,
		ChunkIdx: chunkIdx,
		Layers:   make([][]int, level+1),
	}
	h.nodes = append(h.nodes, n)

	// First element becomes the entry point.
	if h.entryPoint == -1 {
		h.entryPoint = newIdx
		h.maxLayer = level
		return nil
	}

	ep := h.entryPoint
	vs := &visitedSet{}
	// Phase 1: greedily descend from the top down to level+1 with ef=1.
	for lc := h.maxLayer; lc > level; lc-- {
		ep = h.greedyClosest(cp, ep, lc)
	}

	// Phase 2: from min(maxLayer, level) down to 0, search with efConst and
	// wire up bidirectional connections.
	start := h.maxLayer
	if level < start {
		start = level
	}
	for lc := start; lc >= 0; lc-- {
		candidates := h.searchLayer(cp, []int{ep}, h.efConst, lc, vs)
		mmax := h.M
		if lc == 0 {
			mmax = h.Mmax0
		}
		neighbors := h.selectNeighborsHeuristic(cp, candidates, h.M)
		n.Layers[lc] = neighbors

		// Add reverse edges and prune overflowing neighbor lists.
		for _, nb := range neighbors {
			h.nodes[nb].Layers[lc] = append(h.nodes[nb].Layers[lc], newIdx)
			if len(h.nodes[nb].Layers[lc]) > mmax {
				h.nodes[nb].Layers[lc] = h.prune(nb, lc, mmax)
			}
		}
		if len(candidates) > 0 {
			ep = candidates[0].node // nearest, for the next layer down
		}
	}

	if level > h.maxLayer {
		h.maxLayer = level
		h.entryPoint = newIdx
	}
	return nil
}

// greedyClosest walks one layer following the locally closest neighbor until no
// neighbor improves on the current node (ef=1 greedy search).
func (h *HNSW) greedyClosest(q []float32, entry, layer int) int {
	cur := entry
	curDist := h.distance(q, h.nodes[cur].Vec)
	for {
		improved := false
		for _, nb := range h.neighbors(cur, layer) {
			d := h.distance(q, h.nodes[nb].Vec)
			if d < curDist {
				curDist = d
				cur = nb
				improved = true
			}
		}
		if !improved {
			return cur
		}
	}
}

// neighbors returns the neighbor list of a node at a given layer, or nil if the
// node doesn't reach that layer.
func (h *HNSW) neighbors(idx, layer int) []int {
	n := h.nodes[idx]
	if layer >= len(n.Layers) {
		return nil
	}
	return n.Layers[layer]
}

// searchLayer runs the ef-bounded beam search at a single layer, returning the
// best results sorted ascending by distance (closest first). The caller-owned
// visitedSet avoids per-call map allocation, which dominates search cost.
func (h *HNSW) searchLayer(q []float32, entries []int, ef, layer int, vs *visitedSet) []candidate {
	vs.reset(len(h.nodes))
	cand := &minHeap{}
	result := &maxHeap{}

	for _, e := range entries {
		d := h.distance(q, h.nodes[e].Vec)
		vs.set(e)
		heap.Push(cand, candidate{node: e, dist: d})
		heap.Push(result, candidate{node: e, dist: d})
	}

	for cand.Len() > 0 {
		c := heap.Pop(cand).(candidate)
		// If the closest remaining candidate is farther than our worst result,
		// the frontier can't improve the result set.
		if result.Len() > 0 && c.dist > (*result)[0].dist && result.Len() >= ef {
			break
		}
		for _, nb := range h.neighbors(c.node, layer) {
			if vs.test(nb) {
				continue
			}
			vs.set(nb)
			d := h.distance(q, h.nodes[nb].Vec)
			if result.Len() < ef || d < (*result)[0].dist {
				heap.Push(cand, candidate{node: nb, dist: d})
				heap.Push(result, candidate{node: nb, dist: d})
				if result.Len() > ef {
					heap.Pop(result) // drop the farthest
				}
			}
		}
	}

	out := make([]candidate, result.Len())
	for i := len(out) - 1; i >= 0; i-- {
		out[i] = heap.Pop(result).(candidate) // pops farthest first => fill from end
	}
	return out // ascending by distance
}

// selectNeighborsHeuristic implements the HNSW neighbor-selection heuristic
// (Malkov & Yashunin, Algorithm 4). Walking candidates from nearest to the
// reference point outward, it keeps a candidate only if that candidate is
// closer to the reference than to any already-selected neighbor. This favors
// edges that span different "directions", keeping the graph navigable and
// markedly improving recall over naively keeping the m closest. candidates must
// be sorted ascending by dist (distance to ref).
func (h *HNSW) selectNeighborsHeuristic(ref []float32, candidates []candidate, m int) []int {
	R := make([]int, 0, m)
	for _, c := range candidates {
		if len(R) >= m {
			break
		}
		keep := true
		for _, r := range R {
			// If c is nearer to an already-chosen neighbor than to ref, the edge
			// is redundant; skip it.
			if h.distance(h.nodes[c.node].Vec, h.nodes[r].Vec) < c.dist {
				keep = false
				break
			}
		}
		if keep {
			R = append(R, c.node)
		}
	}
	// If the heuristic discarded too many, top up with the closest remaining
	// candidates so we don't under-connect (which would hurt connectivity).
	if len(R) < m {
		inR := make(map[int]bool, len(R))
		for _, r := range R {
			inR[r] = true
		}
		for _, c := range candidates {
			if len(R) >= m {
				break
			}
			if !inR[c.node] {
				R = append(R, c.node)
				inR[c.node] = true
			}
		}
	}
	return R
}

// prune trims a node's neighbor list at a layer back down to mmax using the same
// selection heuristic, with the node itself as the reference point.
func (h *HNSW) prune(idx, layer, mmax int) []int {
	conns := h.nodes[idx].Layers[layer]
	base := h.nodes[idx].Vec
	cands := make([]candidate, len(conns))
	for i, c := range conns {
		cands[i] = candidate{node: c, dist: h.distance(base, h.nodes[c].Vec)}
	}
	sort.Slice(cands, func(i, j int) bool { return cands[i].dist < cands[j].dist })
	return h.selectNeighborsHeuristic(base, cands, mmax)
}

func (h *HNSW) Search(query []float32, k int) ([]SearchResult, error) {
	if h.entryPoint == -1 {
		return nil, nil
	}
	if h.dims != 0 && len(query) != h.dims {
		return nil, fmt.Errorf("hnsw: query dim %d != index dim %d", len(query), h.dims)
	}
	ep := h.entryPoint
	for lc := h.maxLayer; lc > 0; lc-- {
		ep = h.greedyClosest(query, ep, lc)
	}
	ef := h.efSearch
	if ef < k {
		ef = k
	}
	found := h.searchLayer(query, []int{ep}, ef, 0, &visitedSet{})
	if len(found) > k {
		found = found[:k]
	}
	out := make([]SearchResult, len(found))
	for i, c := range found {
		n := h.nodes[c.node]
		out[i] = SearchResult{ID: n.ID, Score: 1 - c.dist, ChunkIdx: n.ChunkIdx}
	}
	return out, nil
}

// hnswGob is the on-disk representation. The graph already uses integer indices
// instead of pointers, so gob can encode it directly.
type hnswGob struct {
	Nodes      []*node
	EntryPoint int
	MaxLayer   int
	M          int
	Mmax0      int
	EfConst    int
	EfSearch   int
	Ml         float64
	Dims       int
}

func (h *HNSW) Save(path string) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()
	enc := gob.NewEncoder(file)
	return enc.Encode(hnswGob{
		Nodes:      h.nodes,
		EntryPoint: h.entryPoint,
		MaxLayer:   h.maxLayer,
		M:          h.M,
		Mmax0:      h.Mmax0,
		EfConst:    h.efConst,
		EfSearch:   h.efSearch,
		Ml:         h.ml,
		Dims:       h.dims,
	})
}

func (h *HNSW) Load(path string) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()
	var g hnswGob
	if err := gob.NewDecoder(file).Decode(&g); err != nil {
		return err
	}
	h.nodes = g.Nodes
	h.entryPoint = g.EntryPoint
	h.maxLayer = g.MaxLayer
	h.M = g.M
	h.Mmax0 = g.Mmax0
	h.efConst = g.EfConst
	h.efSearch = g.EfSearch
	h.ml = g.Ml
	h.dims = g.Dims
	if h.rng == nil {
		h.rng = rand.New(rand.NewSource(1))
	}
	if h.ml == 0 {
		h.ml = 1.0 / math.Log(float64(h.M))
	}
	return nil
}
