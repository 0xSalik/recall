package bootstrap

import (
	"fmt"
	"io"
)

// CLIProgress returns a ProgressFunc that renders a single-line progress bar for
// label to w (typically os.Stderr). Call Done (the returned func) once the
// download finishes to print the terminating newline.
func CLIProgress(w io.Writer, label string) (ProgressFunc, func()) {
	started := false
	render := func(done, total int64) {
		started = true
		if total > 0 {
			pct := float64(done) / float64(total) * 100
			const width = 30
			filled := int(pct / 100 * width)
			bar := make([]byte, width)
			for i := range bar {
				if i < filled {
					bar[i] = '='
				} else {
					bar[i] = ' '
				}
			}
			fmt.Fprintf(w, "\r%s [%s] %5.1f%% (%s / %s)", label, bar, pct, humanBytes(done), humanBytes(total))
		} else {
			fmt.Fprintf(w, "\r%s %s", label, humanBytes(done))
		}
	}
	done := func() {
		if started {
			fmt.Fprintln(w)
		}
	}
	return render, done
}

// humanBytes renders a byte count compactly (e.g. "2.3 GB").
func humanBytes(n int64) string {
	const unit = 1024
	if n < unit {
		return fmt.Sprintf("%d B", n)
	}
	div, exp := int64(unit), 0
	for x := n / unit; x >= unit; x /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %cB", float64(n)/float64(div), "KMGTPE"[exp])
}
