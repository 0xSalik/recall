.PHONY: build test bench wasm clean models

build:
	go build -o bin/recall .

wasm:
	GOOS=js GOARCH=wasm go build -o web/recall.wasm ./web/
	@if [ -f "$$(go env GOROOT)/lib/wasm/wasm_exec.js" ]; then \
		cp "$$(go env GOROOT)/lib/wasm/wasm_exec.js" web/; \
	else \
		cp "$$(go env GOROOT)/misc/wasm/wasm_exec.js" web/; \
	fi

test:
	go test ./...

bench:
	go test -bench=. -benchmem ./internal/...
	go run ./cmd/bench

clean:
	rm -rf bin/ web/recall.wasm

models:
	mkdir -p models
	curl -L -o models/nomic-embed-text-v1.5.Q4_K_M.gguf \
	  https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q4_K_M.gguf
	curl -L -o models/phi-3-mini-4k-instruct.Q4_K_M.gguf \
	  https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf
