package bootstrap

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// withTempHome points UserHomeDir at a temp dir for the duration of a test.
func withTempHome(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	switch runtime.GOOS {
	case "windows":
		t.Setenv("USERPROFILE", dir)
	default:
		t.Setenv("HOME", dir)
	}
	return dir
}

func TestEnsureModelOverride(t *testing.T) {
	withTempHome(t)
	// Existing override file is returned as-is.
	f := filepath.Join(t.TempDir(), "custom.gguf")
	if err := os.WriteFile(f, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	got, err := EnsureEmbedModel(f, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got != f {
		t.Fatalf("override = %q, want %q", got, f)
	}

	// Missing override is an error (no silent download).
	if _, err := EnsureEmbedModel(filepath.Join(t.TempDir(), "nope.gguf"), nil); err == nil {
		t.Fatal("expected error for missing override")
	}
}

func TestEnsureModelUsesCache(t *testing.T) {
	home := withTempHome(t)
	cached := filepath.Join(home, ".recall", "models", DefaultEmbedModel.Filename)
	if err := os.MkdirAll(filepath.Dir(cached), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(cached, []byte("cached-model"), 0o644); err != nil {
		t.Fatal(err)
	}
	// No override, no network: must resolve to the cached copy.
	got, err := EnsureEmbedModel("", nil)
	if err != nil {
		t.Fatal(err)
	}
	if got != cached {
		t.Fatalf("got %q, want cached %q", got, cached)
	}
}

func TestEnsureEngineFallsBackToPath(t *testing.T) {
	withTempHome(t)
	// Nothing cached and (default build) nothing embedded: returns "" so callers
	// fall back to PATH / --bin.
	dir, err := EnsureEngine()
	if err != nil {
		t.Fatal(err)
	}
	if embeddedAvailable() {
		return // bundle build behavior is environment-dependent
	}
	if dir != "" {
		t.Fatalf("expected empty bin dir in default build, got %q", dir)
	}
}

func TestHomeAndDirs(t *testing.T) {
	home := withTempHome(t)
	want := filepath.Join(home, ".recall")
	if Home() != want {
		t.Fatalf("Home() = %q, want %q", Home(), want)
	}
	if filepath.Dir(modelsDir()) != Home() || filepath.Base(modelsDir()) != "models" {
		t.Fatalf("modelsDir() = %q", modelsDir())
	}
}
