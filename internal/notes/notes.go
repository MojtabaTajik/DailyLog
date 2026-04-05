package notes

import (
	"fmt"
	"os"
	"path/filepath"
	"time"
)

// Store encapsulates filesystem persistence for daily markdown notes.
// It abstracts directory layout so the rest of the app can remain
// agnostic of how notes are stored on disk.
type Store struct {
	root string
}

// NewStore returns a Store rooted at the given directory. The directory
// is created lazily when notes are written.
func NewStore(root string) *Store {
	return &Store{root: root}
}

// PathFor returns the absolute path of the note file for the provided date.
// Layout: <root>/<YYYY>/<MM>/<YYYY-MM-DD>.md
func (s *Store) PathFor(t time.Time) string {
	year := fmt.Sprintf("%04d", t.Year())
	month := fmt.Sprintf("%02d", int(t.Month()))
	name := t.Format("2006-01-02") + ".md"
	return filepath.Join(s.root, year, month, name)
}

// Load reads the existing note for a date. If the file does not yet
// exist, an empty string is returned without error.
func (s *Store) Load(t time.Time) (string, error) {
	path := s.PathFor(t)
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return "", nil
		}
		return "", fmt.Errorf("read note: %w", err)
	}
	return string(data), nil
}

// Save writes the given content to the note file for the given date,
// creating any missing parent directories.
func (s *Store) Save(t time.Time, content string) error {
	path := s.PathFor(t)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("create note dir: %w", err)
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		return fmt.Errorf("write note: %w", err)
	}
	return nil
}

// AppendEntry merges a new timestamped entry into the existing content.
// The resulting string is what should be handed to the AI refiner.
func AppendEntry(existing, text string, t time.Time) string {
	header := fmt.Sprintf("## %s", t.Format("15:04"))
	entry := header + "\n" + text + "\n"
	if existing == "" {
		return fmt.Sprintf("# %s\n\n%s", t.Format("2006-01-02"), entry)
	}
	return existing + "\n" + entry
}
