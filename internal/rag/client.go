package rag

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// Client talks to the RAG sidecar over HTTP. The sidecar owns LanceDB
// and the Ollama embedding model; this client is a thin transport.
type Client struct {
	baseURL string
	http    *http.Client
}

// NewClient returns a Client targeting baseURL (e.g. "http://rag:8000").
// Trailing slashes are tolerated.
func NewClient(baseURL string) *Client {
	return &Client{
		baseURL: strings.TrimRight(baseURL, "/"),
		http:    &http.Client{Timeout: 60 * time.Second},
	}
}

type indexRequest struct {
	Key     string `json:"key"`
	Content string `json:"content"`
}

// IndexResult reports what the sidecar did with an /index call.
type IndexResult struct {
	Chunks  int    `json:"chunks"`
	Skipped bool   `json:"skipped"`
	Hash    string `json:"hash"`
}

// Index upserts the chunks for key. The sidecar deletes any prior
// chunks for the same key and re-embeds the new content. If the
// content hash matches what is already indexed, the sidecar skips
// work and returns Skipped=true.
func (c *Client) Index(ctx context.Context, key, content string) (IndexResult, error) {
	var resp IndexResult
	if err := c.post(ctx, "/index", indexRequest{Key: key, Content: content}, &resp); err != nil {
		return IndexResult{}, err
	}
	return resp, nil
}

type queryRequest struct {
	Q string `json:"q"`
	K int    `json:"k"`
}

// Hit is one retrieved chunk plus its source key.
type Hit struct {
	Text    string  `json:"text"`
	Key     string  `json:"key"`
	ChunkID int     `json:"chunk_id"`
	Score   float64 `json:"score"`
}

type queryResponse struct {
	Hits []Hit `json:"hits"`
}

// Query embeds q and returns up to k nearest chunks.
func (c *Client) Query(ctx context.Context, q string, k int) ([]Hit, error) {
	if k <= 0 {
		k = 6
	}
	var resp queryResponse
	if err := c.post(ctx, "/query", queryRequest{Q: q, K: k}, &resp); err != nil {
		return nil, err
	}
	return resp.Hits, nil
}

type statusResponse struct {
	Keys map[string]string `json:"keys"`
}

// Status returns a map of indexed key -> content_hash. Useful for
// callers that want to skip work locally before sending /index.
func (c *Client) Status(ctx context.Context) (map[string]string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+"/status", nil)
	if err != nil {
		return nil, fmt.Errorf("build request: %w", err)
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return nil, fmt.Errorf("rag status: %w", err)
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read status: %w", err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("rag status %d: %s", resp.StatusCode, string(body))
	}
	var parsed statusResponse
	if err := json.Unmarshal(body, &parsed); err != nil {
		return nil, fmt.Errorf("decode status: %w", err)
	}
	return parsed.Keys, nil
}

func (c *Client) post(ctx context.Context, path string, in, out any) error {
	payload, err := json.Marshal(in)
	if err != nil {
		return fmt.Errorf("marshal: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+path, bytes.NewReader(payload))
	if err != nil {
		return fmt.Errorf("build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.http.Do(req)
	if err != nil {
		return fmt.Errorf("rag request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("read response: %w", err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("rag returned %d: %s", resp.StatusCode, string(body))
	}
	if err := json.Unmarshal(body, out); err != nil {
		return fmt.Errorf("decode response: %w", err)
	}
	return nil
}
