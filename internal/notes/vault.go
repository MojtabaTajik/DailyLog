package notes

import (
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
)

// skippedDirs are directories that Walk never descends into. They are
// matched on the directory name only (any depth). Hidden dirs starting
// with "." (.git, .obsidian, .trash, .claude, .vscode, ...) are also
// skipped uniformly.
var skippedDirs = map[string]struct{}{
	"node_modules": {},
}

// VaultStore is a read-only walker over an Obsidian-style vault. It is
// independent of the daily-notes Store so the daily-notes write path
// stays unchanged while the RAG layer indexes the entire vault.
type VaultStore struct {
	root string
}

// NewVaultStore returns a walker rooted at root. An empty root yields
// a walker whose Walk is a no-op, which lets callers stay simple when
// VAULT_PATH is not configured.
func NewVaultStore(root string) *VaultStore {
	return &VaultStore{root: root}
}

// Root returns the configured vault root.
func (v *VaultStore) Root() string { return v.root }

// VaultEntry is one markdown note discovered by Walk.
type VaultEntry struct {
	// Key is the path of the file relative to the vault root, using
	// forward slashes. It is suitable as an opaque identifier for the
	// RAG sidecar's /index call.
	Key string
	// AbsPath is the absolute path of the file on disk.
	AbsPath string
	// Content is the file contents as a string.
	Content string
}

// VaultWalkFunc is invoked for every markdown note found.
type VaultWalkFunc func(VaultEntry) error

// Walk visits every *.md file in the vault, skipping hidden dirs and
// well-known auxiliary dirs. Files are visited in lexical path order
// for predictable progress reporting.
func (v *VaultStore) Walk(fn VaultWalkFunc) error {
	if v.root == "" {
		return nil
	}
	if _, err := os.Stat(v.root); os.IsNotExist(err) {
		return nil
	}

	var paths []string
	err := filepath.WalkDir(v.root, func(path string, d fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		name := d.Name()
		if d.IsDir() {
			if path == v.root {
				return nil
			}
			if strings.HasPrefix(name, ".") {
				return filepath.SkipDir
			}
			if _, skip := skippedDirs[name]; skip {
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.EqualFold(filepath.Ext(name), ".md") {
			return nil
		}
		paths = append(paths, path)
		return nil
	})
	if err != nil {
		return fmt.Errorf("walk vault: %w", err)
	}

	for _, p := range paths {
		key, err := v.KeyFor(p)
		if err != nil {
			return err
		}
		data, err := os.ReadFile(p)
		if err != nil {
			return fmt.Errorf("read %s: %w", p, err)
		}
		if err := fn(VaultEntry{Key: key, AbsPath: p, Content: string(data)}); err != nil {
			return err
		}
	}
	return nil
}

// KeyFor returns the vault-relative key for an absolute path inside
// the vault. Returns an error if absPath is outside the vault root.
func (v *VaultStore) KeyFor(absPath string) (string, error) {
	if v.root == "" {
		return "", fmt.Errorf("vault root not configured")
	}
	rel, err := filepath.Rel(v.root, absPath)
	if err != nil {
		return "", fmt.Errorf("relative path: %w", err)
	}
	if strings.HasPrefix(rel, "..") {
		return "", fmt.Errorf("path %q is outside vault %q", absPath, v.root)
	}
	return filepath.ToSlash(rel), nil
}
