package bootstrap

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
)

// Home is the recall home directory (~/.recall), where the store, downloaded
// models, and extracted engine binaries live. It falls back to ./.recall when
// the user's home directory cannot be determined.
func Home() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return ".recall"
	}
	return filepath.Join(home, ".recall")
}

func modelsDir() string { return filepath.Join(Home(), "models") }
func binDir() string     { return filepath.Join(Home(), "bin") }

// exeSuffix is the platform executable extension.
func exeSuffix() string {
	if runtime.GOOS == "windows" {
		return ".exe"
	}
	return ""
}

// EnsureEmbedModel resolves the embedding model, returning a usable path.
func EnsureEmbedModel(override string, progress ProgressFunc) (string, error) {
	return ensureModel(DefaultEmbedModel, override, progress)
}

// EnsureGenModel resolves the generation model, returning a usable path.
func EnsureGenModel(override string, progress ProgressFunc) (string, error) {
	return ensureModel(DefaultGenModel, override, progress)
}

// ensureModel applies the resolution order: explicit override -> on-disk cache
// -> embedded copy -> download.
func ensureModel(m Model, override string, progress ProgressFunc) (string, error) {
	if override != "" {
		if fileExists(override) {
			return override, nil
		}
		return "", fmt.Errorf("bootstrap: model %q not found", override)
	}

	dst := filepath.Join(modelsDir(), m.Filename)
	if fileExists(dst) {
		return dst, nil
	}
	if err := os.MkdirAll(modelsDir(), 0o755); err != nil {
		return "", fmt.Errorf("bootstrap: creating models dir: %w", err)
	}

	if rc, ok := embeddedOpen("models/" + m.Filename); ok {
		defer rc.Close()
		if err := writeFile(dst, rc, 0o644); err != nil {
			return "", err
		}
		return dst, nil
	}

	if m.URL == "" {
		return "", fmt.Errorf("bootstrap: no source available for %s", m.Filename)
	}
	if err := download(dst, m, progress); err != nil {
		return "", err
	}
	return dst, nil
}

// EnsureEngine makes the llama.cpp binaries available. It returns a directory to
// prepend to PATH so the binaries are discoverable, or "" when none is bundled
// or cached (in which case recall relies on the system PATH or the --bin flag,
// the historical behavior).
func EnsureEngine() (string, error) {
	bd := binDir()
	if engineCached(bd) {
		return bd, nil
	}
	if embeddedAvailable() {
		if err := extractEngine(bd); err != nil {
			return "", err
		}
		if engineCached(bd) {
			return bd, nil
		}
	}
	return "", nil
}

// engineCached reports whether every engine binary is present in dir.
func engineCached(dir string) bool {
	for _, name := range engineBinaries {
		if !fileExists(filepath.Join(dir, name+exeSuffix())) {
			return false
		}
	}
	return true
}

// extractEngine writes the embedded engine binaries into dir with exec perms.
func extractEngine(dir string) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("bootstrap: creating bin dir: %w", err)
	}
	for _, name := range engineBinaries {
		fname := name + exeSuffix()
		rc, ok := embeddedOpen("bin/" + fname)
		if !ok {
			// Not all binaries are necessarily bundled; skip missing ones and
			// let engineCached decide whether the result is usable.
			continue
		}
		err := writeFile(filepath.Join(dir, fname), rc, 0o755)
		rc.Close()
		if err != nil {
			return err
		}
	}
	return nil
}

func fileExists(path string) bool {
	info, err := os.Stat(path)
	return err == nil && !info.IsDir()
}

// writeFile copies r into path atomically (via a temp file + rename) with the
// given permissions.
func writeFile(path string, r io.Reader, perm os.FileMode) (err error) {
	tmp := path + ".tmp"
	f, err := os.OpenFile(tmp, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, perm)
	if err != nil {
		return fmt.Errorf("bootstrap: creating %s: %w", tmp, err)
	}
	if _, err = io.Copy(f, r); err != nil {
		f.Close()
		os.Remove(tmp)
		return fmt.Errorf("bootstrap: writing %s: %w", tmp, err)
	}
	if err = f.Close(); err != nil {
		os.Remove(tmp)
		return err
	}
	if err = os.Chmod(tmp, perm); err != nil {
		os.Remove(tmp)
		return err
	}
	if err = os.Rename(tmp, path); err != nil {
		os.Remove(tmp)
		return fmt.Errorf("bootstrap: finalizing %s: %w", path, err)
	}
	return nil
}
