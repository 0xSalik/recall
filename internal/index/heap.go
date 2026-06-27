package index

// candidate pairs a node index with its distance to a query. HNSW search keeps
// two ordered sets of these: a min-heap of candidates to explore and a max-heap
// of the best results found so far.
type candidate struct {
	node int
	dist float32
}

// minHeap orders candidates by ascending distance (closest at the root). It is
// the frontier of nodes still to be explored.
type minHeap []candidate

func (h minHeap) Len() int           { return len(h) }
func (h minHeap) Less(i, j int) bool { return h[i].dist < h[j].dist }
func (h minHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *minHeap) Push(x any)        { *h = append(*h, x.(candidate)) }
func (h *minHeap) Pop() any {
	old := *h
	n := len(old)
	it := old[n-1]
	*h = old[:n-1]
	return it
}

// maxHeap orders candidates by descending distance (farthest at the root) so we
// can cheaply evict the worst result when the result set is full.
type maxHeap []candidate

func (h maxHeap) Len() int           { return len(h) }
func (h maxHeap) Less(i, j int) bool { return h[i].dist > h[j].dist }
func (h maxHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *maxHeap) Push(x any)        { *h = append(*h, x.(candidate)) }
func (h *maxHeap) Pop() any {
	old := *h
	n := len(old)
	it := old[n-1]
	*h = old[:n-1]
	return it
}
