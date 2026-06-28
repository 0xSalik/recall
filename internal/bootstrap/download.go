package bootstrap

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"time"
)

// ProgressFunc receives download progress. done is bytes written so far; total
// is the expected total (0 if unknown). It may be called frequently; keep it
// cheap. It is safe for it to be nil at the call site (see download).
type ProgressFunc func(done, total int64)

// httpClient is shared; model downloads are large so no overall timeout (only a
// dial/idle safety net would go here if needed).
var httpClient = &http.Client{Timeout: 0}

// download fetches m.URL into dst. It writes to dst+".part", resumes a previous
// partial download via a Range request when possible, verifies the SHA-256 when
// m.SHA256 is set, and atomically renames into place on success. progress may be
// nil.
func download(dst string, m Model, progress ProgressFunc) (err error) {
	part := dst + ".part"

	// Resume: if a partial file exists, continue from its current size.
	var existing int64
	if fi, statErr := os.Stat(part); statErr == nil {
		existing = fi.Size()
	}

	req, err := http.NewRequest(http.MethodGet, m.URL, nil)
	if err != nil {
		return fmt.Errorf("bootstrap: building request: %w", err)
	}
	if existing > 0 {
		req.Header.Set("Range", "bytes="+strconv.FormatInt(existing, 10)+"-")
	}

	resp, err := httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("bootstrap: downloading %s: %w", m.Filename, err)
	}
	defer resp.Body.Close()

	// Decide append vs truncate based on whether the server honored the range.
	flag := os.O_CREATE | os.O_WRONLY | os.O_TRUNC
	resumed := false
	switch resp.StatusCode {
	case http.StatusOK:
		existing = 0 // server ignored Range; start over
	case http.StatusPartialContent:
		flag = os.O_CREATE | os.O_WRONLY | os.O_APPEND
		resumed = true
	default:
		return fmt.Errorf("bootstrap: downloading %s: unexpected status %s", m.Filename, resp.Status)
	}

	total := m.Size
	if cl := resp.ContentLength; cl > 0 {
		if resumed {
			total = existing + cl
		} else {
			total = cl
		}
	}

	f, err := os.OpenFile(part, flag, 0o644)
	if err != nil {
		return fmt.Errorf("bootstrap: opening %s: %w", part, err)
	}
	defer func() {
		cerr := f.Close()
		if err == nil {
			err = cerr
		}
	}()

	pw := &progressWriter{w: f, done: existing, total: total, fn: progress}
	if _, err = io.Copy(pw, resp.Body); err != nil {
		return fmt.Errorf("bootstrap: writing %s: %w", part, err)
	}

	if m.SHA256 != "" {
		sum, herr := fileSHA256(part)
		if herr != nil {
			return herr
		}
		if sum != m.SHA256 {
			os.Remove(part)
			return fmt.Errorf("bootstrap: checksum mismatch for %s: got %s, want %s", m.Filename, sum, m.SHA256)
		}
	}

	if err = os.Rename(part, dst); err != nil {
		return fmt.Errorf("bootstrap: finalizing %s: %w", dst, err)
	}
	return nil
}

func fileSHA256(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// progressWriter wraps a writer and reports cumulative progress, throttled so a
// terminal-bound callback isn't invoked on every chunk.
type progressWriter struct {
	w     io.Writer
	done  int64
	total int64
	fn    ProgressFunc
	last  time.Time
}

func (p *progressWriter) Write(b []byte) (int, error) {
	n, err := p.w.Write(b)
	p.done += int64(n)
	if p.fn != nil && (time.Since(p.last) > 100*time.Millisecond) {
		p.fn(p.done, p.total)
		p.last = time.Now()
	}
	return n, err
}
