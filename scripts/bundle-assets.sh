#!/usr/bin/env bash
# bundle-assets.sh — populate internal/bootstrap/assets/ for a "bundle" build.
#
# It builds the CPU-only llama.cpp engine binaries from source (static where the
# platform allows) and downloads the embedding model, placing both where the
# `bundle` build tag embeds them:
#
#   internal/bootstrap/assets/bin/llama-embedding[.exe]
#   internal/bootstrap/assets/bin/llama-completion[.exe]
#   internal/bootstrap/assets/models/<embedding model>.gguf
#
# The large *generation* model is intentionally NOT bundled — recall downloads it
# on first use to keep the release binary under GitHub's 2 GB asset limit.
#
# Usage:
#   scripts/bundle-assets.sh            # build engine + fetch embed model
#   LLAMA_REF=b4000 scripts/bundle-assets.sh   # pin a llama.cpp tag
#
# Requires: git, cmake, a C/C++ toolchain, curl.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSETS="$ROOT/internal/bootstrap/assets"
BIN_DIR="$ASSETS/bin"
MODELS_DIR="$ASSETS/models"
LLAMA_REF="${LLAMA_REF:-master}"
EMBED_URL="https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q4_K_M.gguf?download=true"
EMBED_FILE="nomic-embed-text-v1.5.Q4_K_M.gguf"

EXE=""
case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*) EXE=".exe" ;;
esac

mkdir -p "$BIN_DIR" "$MODELS_DIR"

echo ">> Building llama.cpp ($LLAMA_REF), CPU-only"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
git clone --depth 1 --branch "$LLAMA_REF" https://github.com/ggml-org/llama.cpp "$WORK/llama.cpp" 2>/dev/null \
  || git clone --depth 1 https://github.com/ggml-org/llama.cpp "$WORK/llama.cpp"

# GGML_NATIVE=OFF: do NOT compile for the build machine's exact CPU (the CI
# runner often has AVX-512); a -march=native binary would crash with "illegal
# instruction" on users' older CPUs. OFF builds a portable baseline.
cmake -S "$WORK/llama.cpp" -B "$WORK/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL=OFF -DGGML_CUDA=OFF -DGGML_VULKAN=OFF \
  -DGGML_NATIVE=OFF \
  -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=OFF

# Cap parallelism: current llama.cpp compiles many large translation units
# (src/models/*.cpp), and an uncapped -j exhausts RAM on the memory-limited CI
# runners, which the kernel OOM-kills (the build dies at ~80% and the runner is
# torn down). JOBS defaults to a conservative 2; override for faster local builds.
JOBS="${JOBS:-2}"
cmake --build "$WORK/build" --config Release -j "$JOBS" --target llama-embedding llama-completion

# Binary names/locations have drifted across llama.cpp versions; search for them.
for tool in llama-embedding llama-completion; do
  found="$(find "$WORK/build" -type f -name "${tool}${EXE}" -perm -u+x 2>/dev/null | head -n1 || true)"
  [ -z "$found" ] && found="$(find "$WORK/build" -type f -name "${tool}${EXE}" 2>/dev/null | head -n1 || true)"
  if [ -z "$found" ]; then
    echo "!! could not locate built $tool — your llama.cpp ref may name it differently" >&2
    exit 1
  fi
  cp "$found" "$BIN_DIR/${tool}${EXE}"
  chmod +x "$BIN_DIR/${tool}${EXE}"
  echo "   bundled $tool"
done

echo ">> Fetching embedding model"
if [ ! -f "$MODELS_DIR/$EMBED_FILE" ]; then
  curl -fL --retry 3 -o "$MODELS_DIR/$EMBED_FILE" "$EMBED_URL"
fi
echo "   $MODELS_DIR/$EMBED_FILE"

echo ">> Assets ready. Build with: go build -tags bundle -o recall ."
