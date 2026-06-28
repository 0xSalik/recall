package bootstrap

import (
	"crypto/sha256"
	"encoding/hex"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

func TestDownloadFullWithChecksum(t *testing.T) {
	payload := []byte(strings.Repeat("recall-model-bytes ", 1000))
	sum := sha256.Sum256(payload)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.ServeContent(w, r, "model.gguf", testModTime(), strings.NewReader(string(payload)))
	}))
	defer srv.Close()

	dst := filepath.Join(t.TempDir(), "model.gguf")
	m := Model{Filename: "model.gguf", URL: srv.URL, SHA256: hex.EncodeToString(sum[:]), Size: int64(len(payload))}

	var lastDone int64
	if err := download(dst, m, func(done, total int64) { lastDone = done }); err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile(dst)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != string(payload) {
		t.Fatal("downloaded content mismatch")
	}
	if lastDone == 0 {
		t.Error("progress callback never reported bytes")
	}
	if _, err := os.Stat(dst + ".part"); !os.IsNotExist(err) {
		t.Error(".part file should be gone after success")
	}
}

func TestDownloadChecksumMismatchRejected(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("the wrong bytes"))
	}))
	defer srv.Close()

	dst := filepath.Join(t.TempDir(), "model.gguf")
	m := Model{Filename: "model.gguf", URL: srv.URL, SHA256: hex.EncodeToString(make([]byte, 32))}
	if err := download(dst, m, nil); err == nil {
		t.Fatal("expected checksum mismatch error")
	}
	if _, err := os.Stat(dst); !os.IsNotExist(err) {
		t.Error("file with bad checksum must not be finalized")
	}
}

func TestDownloadResumesPartial(t *testing.T) {
	payload := []byte(strings.Repeat("0123456789", 500))
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Honor Range so we can assert resume behavior.
		http.ServeContent(w, r, "m.gguf", testModTime(), strings.NewReader(string(payload)))
	}))
	defer srv.Close()

	dir := t.TempDir()
	dst := filepath.Join(dir, "m.gguf")

	// Pre-seed a partial download with the correct prefix bytes.
	half := len(payload) / 2
	if err := os.WriteFile(dst+".part", payload[:half], 0o644); err != nil {
		t.Fatal(err)
	}

	sum := sha256.Sum256(payload)
	m := Model{Filename: "m.gguf", URL: srv.URL, SHA256: hex.EncodeToString(sum[:])}
	if err := download(dst, m, nil); err != nil {
		t.Fatal(err)
	}
	got, _ := os.ReadFile(dst)
	if string(got) != string(payload) {
		t.Fatalf("resumed download produced wrong content (len %d, want %d)", len(got), len(payload))
	}
}

func TestDownloadRangeServerSendsPartial(t *testing.T) {
	// A minimal server that asserts it received a Range header on resume.
	full := strings.Repeat("x", 2048)
	var sawRange bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if rng := r.Header.Get("Range"); rng != "" {
			sawRange = true
			spec := strings.TrimSuffix(strings.TrimPrefix(rng, "bytes="), "-")
			start, _ := strconv.Atoi(spec)
			w.Header().Set("Content-Range", "bytes "+strconv.Itoa(start)+"-"+strconv.Itoa(len(full)-1)+"/"+strconv.Itoa(len(full)))
			w.WriteHeader(http.StatusPartialContent)
			w.Write([]byte(full[start:]))
			return
		}
		w.Write([]byte(full))
	}))
	defer srv.Close()

	dir := t.TempDir()
	dst := filepath.Join(dir, "f.bin")
	os.WriteFile(dst+".part", []byte(full[:1000]), 0o644)

	if err := download(dst, Model{Filename: "f.bin", URL: srv.URL}, nil); err != nil {
		t.Fatal(err)
	}
	if !sawRange {
		t.Error("expected a Range request on resume")
	}
	got, _ := os.ReadFile(dst)
	if string(got) != full {
		t.Fatalf("content mismatch after partial resume: len %d", len(got))
	}
}

func testModTime() time.Time { return time.Unix(1700000000, 0) }
