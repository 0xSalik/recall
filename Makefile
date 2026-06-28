.PHONY: build build-ui bundle bundle-assets test bench clean models

build: build-ui
	go build -o bin/recall .

# Self-contained release build: embed the engine binaries + embedding model so a
# downloaded binary runs with no setup. The generation model is fetched on first
# use (kept out of the binary to stay under GitHub's 2 GB asset limit).
bundle-assets:
	bash scripts/bundle-assets.sh

bundle: build-ui bundle-assets
	go build -tags bundle -o bin/recall .
	@echo "built self-contained bin/recall (engine + embed model embedded)"

# The browser UI is a single static file embedded via go:embed (see web/web.go).
# There is no separate frontend build step; this target validates that the
# embedded asset is present so `go build` will succeed.
build-ui:
	@test -f web/index.html || { echo "web/index.html missing"; exit 1; }
	@echo "ui: web/index.html ready (embedded at build time)"

# Scoped to our own packages so an in-repo third_party/llama.cpp clone (which the
# go tool would otherwise walk) doesn't pollute the output.
test:
	go test ./internal/... ./cmd/... .

bench:
	go test -bench=. -benchmem ./internal/...
	go run ./cmd/bench

clean:
	rm -rf bin/

models:
	mkdir -p models
	curl -L -o models/nomic-embed-text-v1.5.Q4_K_M.gguf \
	  https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q4_K_M.gguf
	curl -L -o models/phi-3-mini-4k-instruct.Q4_K_M.gguf \
	  https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf
