// Package bootstrap makes recall self-provisioning: it resolves the llama.cpp
// engine binaries and the GGUF model files a command needs, transparently
// extracting copies embedded in the binary (in "bundle" builds) or downloading
// them on first use into the recall home directory. This is what lets a freshly
// downloaded binary run without the user manually fetching models or llama.cpp.
//
// Resolution order for a model is always: explicit flag override -> on-disk
// cache (~/.recall/models) -> embedded copy (bundle builds) -> download.
package bootstrap

// Model describes a downloadable GGUF model: where to get it and how to verify
// it. SHA256 is optional but strongly recommended — when set, a downloaded file
// is rejected unless its digest matches, which guards against truncated or
// tampered downloads of code-adjacent weight files.
type Model struct {
	// Filename is the local name under ~/.recall/models and, in bundle builds,
	// the name under the embedded models directory.
	Filename string
	// URL is the canonical download location (e.g. a pinned Hugging Face blob).
	URL string
	// SHA256 is the lowercase hex digest of the file. Empty means "unverified".
	SHA256 string
	// Size is the expected byte size, used only to render download progress.
	Size int64
}

// The default models. These mirror the historical defaults in cmd. The embedding
// model is small (~84 MB) and is embedded in bundle builds; the generation model
// is large and is always downloaded on first use (Option C).
//
// NOTE: fill in SHA256 (and Size) before publishing a release. They are left
// empty here because the digests must be taken from the exact pinned artifacts;
// downloads still work without them, just unverified.
var (
	DefaultEmbedModel = Model{
		Filename: "nomic-embed-text-v1.5.Q4_K_M.gguf",
		URL:      "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q4_K_M.gguf?download=true",
		SHA256:   "",
		Size:     84106624,
	}
	DefaultGenModel = Model{
		Filename: "Phi-3-mini-4k-instruct-q4.gguf",
		URL:      "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf?download=true",
		SHA256:   "",
		Size:     2393231072,
	}
)

// engineBinaries are the llama.cpp executables recall invokes. In bundle builds
// these are embedded and extracted to ~/.recall/bin; otherwise they are expected
// on PATH (or via --bin). Names are without the platform executable suffix; the
// .exe suffix is added on Windows during extraction/lookup.
var engineBinaries = []string{"llama-embedding", "llama-completion"}
