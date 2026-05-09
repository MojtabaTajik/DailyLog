package groq

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"strings"
	"time"
)

const (
	chatEndpoint       = "https://api.groq.com/openai/v1/chat/completions"
	transcribeEndpoint = "https://api.groq.com/openai/v1/audio/transcriptions"
)

// Client is a minimal Groq client for chat completions and audio
// transcriptions. Groq exposes OpenAI-compatible endpoints, so we
// hand-roll the requests to keep the dependency surface tiny.
type Client struct {
	apiKey          string
	model           string
	transcribeModel string
	systemPrompt    string
	http            *http.Client
}

// NewClient constructs a Groq client with sensible HTTP timeouts.
func NewClient(apiKey, model, transcribeModel, systemPrompt string) *Client {
	return &Client{
		apiKey:          apiKey,
		model:           model,
		transcribeModel: transcribeModel,
		systemPrompt:    systemPrompt,
		http:            &http.Client{Timeout: 120 * time.Second},
	}
}

type message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatRequest struct {
	Model    string    `json:"model"`
	Messages []message `json:"messages"`
}

type chatResponse struct {
	Choices []struct {
		Message message `json:"message"`
	} `json:"choices"`
	Error *struct {
		Message string `json:"message"`
		Type    string `json:"type"`
	} `json:"error,omitempty"`
}

// Refine sends the raw markdown to Groq and returns the refined output.
func (c *Client) Refine(ctx context.Context, rawMarkdown string) (string, error) {
	return c.chat(ctx, c.systemPrompt, rawMarkdown)
}

const answerSystemPrompt = `You are a personal-notes assistant answering questions about the user's own Obsidian vault.
You will be given a question and a set of CONTEXT excerpts from the user's notes. Each excerpt may begin with a "Date: YYYY-MM-DD (Day)" line and a "Section: ..." line; treat those as authoritative metadata for the bullet text that follows.

Rules:
1. Answer using ONLY the provided context. If the answer is not in the context, say you could not find it in the notes.
2. Be concise. One short paragraph or a few bullets.
3. When the user asks about timing, dates, weeks, or "when", cite the relevant date(s) from the context (e.g. "On 2025-05-08 you..."). Do not cite file paths.
4. Never invent details that are not in the context.
5. Reply in the same language the question is asked in.`

// Answer runs a retrieval-augmented chat: the question plus a formatted
// context block are sent to the same chat model. Caller supplies the
// already-formatted context (date headers + chunk text).
func (c *Client) Answer(ctx context.Context, question, contextBlock string) (string, error) {
	user := "CONTEXT:\n" + contextBlock + "\n\nQUESTION: " + question
	return c.chat(ctx, answerSystemPrompt, user)
}

const rewriteSystemPrompt = `You rewrite a user's question into a small JSON envelope used by a personal-notes search system.

Output strictly a JSON object of the form:
{"queries": ["..."], "date_from": "YYYY-MM-DD", "date_to": "YYYY-MM-DD"}

Rules:
- "queries": 1 to 3 short keyword-focused search strings derived from the question. Expand vague pronouns ("the thing", "that one") into the most likely concrete noun. Spell out abbreviations. Drop filler words. Keep proper nouns and project names verbatim.
- "date_from" / "date_to": inclusive ISO range if the user clearly references a time period; otherwise empty strings. Do not invent a range.
- Reply with the JSON object only. No preamble, no markdown, no code fences.`

// Rewrite asks the model to expand a terse question into search-ready
// strings and an optional date range. Returns scalars rather than a
// struct so the bot's Rewriter interface stays decoupled from this
// package.
func (c *Client) Rewrite(ctx context.Context, question string) (queries []string, dateFrom, dateTo string, err error) {
	raw, err := c.chat(ctx, rewriteSystemPrompt, question)
	if err != nil {
		return nil, "", "", err
	}
	cleaned := strings.TrimSpace(raw)
	cleaned = strings.TrimPrefix(cleaned, "```json")
	cleaned = strings.TrimPrefix(cleaned, "```")
	cleaned = strings.TrimSuffix(cleaned, "```")
	cleaned = strings.TrimSpace(cleaned)

	var parsed struct {
		Queries  []string `json:"queries"`
		DateFrom string   `json:"date_from"`
		DateTo   string   `json:"date_to"`
	}
	if err := json.Unmarshal([]byte(cleaned), &parsed); err != nil {
		return nil, "", "", fmt.Errorf("decode rewrite: %w (raw=%q)", err, raw)
	}
	return parsed.Queries, parsed.DateFrom, parsed.DateTo, nil
}

func (c *Client) chat(ctx context.Context, system, user string) (string, error) {
	reqBody := chatRequest{
		Model: c.model,
		Messages: []message{
			{Role: "system", Content: system},
			{Role: "user", Content: user},
		},
	}

	payload, err := json.Marshal(reqBody)
	if err != nil {
		return "", fmt.Errorf("marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, chatEndpoint, bytes.NewReader(payload))
	if err != nil {
		return "", fmt.Errorf("build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.apiKey)

	resp, err := c.http.Do(req)
	if err != nil {
		return "", fmt.Errorf("groq request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("read groq response: %w", err)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return "", fmt.Errorf("groq returned %d: %s", resp.StatusCode, string(body))
	}

	var parsed chatResponse
	if err := json.Unmarshal(body, &parsed); err != nil {
		return "", fmt.Errorf("decode groq response: %w", err)
	}
	if parsed.Error != nil {
		return "", fmt.Errorf("groq error: %s", parsed.Error.Message)
	}
	if len(parsed.Choices) == 0 {
		return "", fmt.Errorf("groq returned no choices")
	}
	return parsed.Choices[0].Message.Content, nil
}

// Transcribe sends an audio stream to Groq's Whisper endpoint and
// returns the plain-text transcription. filename must carry an
// extension that Whisper recognizes (e.g. ".ogg" for Telegram voice
// notes), otherwise the API rejects the upload.
func (c *Client) Transcribe(ctx context.Context, audio io.Reader, filename string) (string, error) {
	var body bytes.Buffer
	mw := multipart.NewWriter(&body)

	fileWriter, err := mw.CreateFormFile("file", filename)
	if err != nil {
		return "", fmt.Errorf("create form file: %w", err)
	}
	if _, err := io.Copy(fileWriter, audio); err != nil {
		return "", fmt.Errorf("copy audio: %w", err)
	}
	if err := mw.WriteField("model", c.transcribeModel); err != nil {
		return "", fmt.Errorf("write model field: %w", err)
	}
	// "text" returns the raw transcript as the response body, saving
	// us a JSON decode step.
	if err := mw.WriteField("response_format", "text"); err != nil {
		return "", fmt.Errorf("write response_format field: %w", err)
	}
	if err := mw.Close(); err != nil {
		return "", fmt.Errorf("close multipart: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, transcribeEndpoint, &body)
	if err != nil {
		return "", fmt.Errorf("build transcribe request: %w", err)
	}
	req.Header.Set("Content-Type", mw.FormDataContentType())
	req.Header.Set("Authorization", "Bearer "+c.apiKey)

	resp, err := c.http.Do(req)
	if err != nil {
		return "", fmt.Errorf("groq transcribe request: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("read transcribe response: %w", err)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return "", fmt.Errorf("groq transcribe returned %d: %s", resp.StatusCode, string(respBody))
	}

	return string(bytes.TrimSpace(respBody)), nil
}
