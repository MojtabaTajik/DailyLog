package bot

import (
	"context"
	"fmt"
	"io"
	"log"
	"strings"
	"sync"
	"time"

	"github.com/mojix/dailylog/internal/config"
	"github.com/mojix/dailylog/internal/notes"
	"github.com/mojix/dailylog/internal/rag"
	tele "gopkg.in/telebot.v3"
)

const (
	previewLimit      = 200
	groqTimeout       = 90 * time.Second
	transcribeTimeout = 120 * time.Second
	pollingTimeout    = 10 * time.Second
	pendingTTL        = 30 * time.Minute
	cleanupInterval   = 5 * time.Minute
	ragIndexTimeout   = 60 * time.Second
	ragQueryTimeout   = 30 * time.Second
	ragTopK           = 6
	queryPrefix       = "?"
)

// Inline buttons attached to each note prompt. The Unique field is what
// telebot uses to route the callback back to the correct handler.
var (
	btnYesterday = tele.Btn{Unique: "note_yesterday", Text: "Yesterday"}
	btnToday     = tele.Btn{Unique: "note_today", Text: "Today"}
)

// Refiner is the subset of the Groq client used by the bot for daily
// note refinement. Kept as an interface so the bot stays decoupled from
// the concrete transport.
type Refiner interface {
	Refine(ctx context.Context, raw string) (string, error)
}

// Answerer runs retrieval-augmented question answering against an LLM.
type Answerer interface {
	Answer(ctx context.Context, question, contextBlock string) (string, error)
}

// RagClient is the subset of the RAG sidecar used by the bot.
type RagClient interface {
	Index(ctx context.Context, date, content string) (rag.IndexResult, error)
	Query(ctx context.Context, q string, k int) ([]rag.Hit, error)
	Status(ctx context.Context) (map[string]string, error)
}

// Transcriber converts an audio stream to text. filename must carry a
// Whisper-recognized extension (e.g. ".ogg" for Telegram voice notes).
type Transcriber interface {
	Transcribe(ctx context.Context, audio io.Reader, filename string) (string, error)
}

// NoteStore is the persistence contract for the daily-notes write path.
type NoteStore interface {
	Load(t time.Time) (string, error)
	Save(t time.Time, content string) error
	PathFor(t time.Time) string
}

// Vault is the read-only walker over the full Obsidian vault used by
// the RAG indexing path.
type Vault interface {
	Walk(fn notes.VaultWalkFunc) error
	KeyFor(absPath string) (string, error)
	Root() string
}

// pendingNote holds a note awaiting the user's day-selection click.
// reactTarget is the message that receives status reactions — for voice
// notes this is the transcription reply, for text notes it's the user's
// message itself.
type pendingNote struct {
	text        string
	reactTarget *tele.Message
	createdAt   time.Time
}

// Bot wires together Telegram, the note store, and the AI refiner.
type Bot struct {
	cfg         *config.Config
	tele        *tele.Bot
	store       NoteStore
	vault       Vault
	refiner     Refiner
	answerer    Answerer
	transcriber Transcriber
	rag         RagClient // may be nil if RAG is not configured

	pendingMu sync.Mutex
	pending   map[int]*pendingNote
}

// New constructs a Bot and registers handlers. It returns an error if
// the underlying Telegram client cannot be initialized. ragClient may
// be nil to disable indexing and the ? query path.
func New(cfg *config.Config, store NoteStore, vault Vault, refiner Refiner, answerer Answerer, transcriber Transcriber, ragClient RagClient) (*Bot, error) {
	settings := tele.Settings{
		Token:  cfg.TelegramToken,
		Poller: &tele.LongPoller{Timeout: pollingTimeout},
	}

	tb, err := tele.NewBot(settings)
	if err != nil {
		return nil, fmt.Errorf("telebot init: %w", err)
	}

	b := &Bot{
		cfg:         cfg,
		tele:        tb,
		store:       store,
		vault:       vault,
		refiner:     refiner,
		answerer:    answerer,
		transcriber: transcriber,
		rag:         ragClient,
		pending:     make(map[int]*pendingNote),
	}
	b.registerHandlers()
	go b.cleanupLoop()
	return b, nil
}

// Start blocks and runs the long-polling loop.
func (b *Bot) Start() {
	b.tele.Start()
}

func (b *Bot) registerHandlers() {
	b.tele.Use(b.onlyAuthorizedChat)

	// This bot is dedicated to daily notes: every plain text message is
	// treated as a note to append. /help and /start remain as explicit
	// commands so the user can always discover what the bot does.
	b.tele.Handle("/help", b.handleHelp)
	b.tele.Handle("/start", b.handleHelp)
	b.tele.Handle("/reindex", b.handleReindex)
	b.tele.Handle(tele.OnText, b.handleDaily)
	b.tele.Handle(tele.OnVoice, b.handleVoice)

	b.tele.Handle(&btnToday, b.handleDayChoice(0))
	b.tele.Handle(&btnYesterday, b.handleDayChoice(-1))
}

// onlyAuthorizedChat is middleware that drops any update whose chat ID
// does not match the configured allow-listed chat.
func (b *Bot) onlyAuthorizedChat(next tele.HandlerFunc) tele.HandlerFunc {
	return func(c tele.Context) error {
		if c.Chat() == nil || c.Chat().ID != b.cfg.TelegramChatID {
			log.Printf("ignoring update from unauthorized chat: %v", c.Chat())
			return nil
		}
		return next(c)
	}
}

func (b *Bot) handleHelp(c tele.Context) error {
	return c.Send(strings.Join([]string{
		"dailylog bot:",
		"Send a text message or a voice note and the bot will ask whether to file it under Today or Yesterday, then append and refine it.",
		"Start a message with ? to ask a question about your past notes (e.g. \"? when did I fix the kitchen tap?\").",
		"/reindex: re-embed all past notes into the search index (skips unchanged days).",
		"/help: show this message",
	}, "\n"))
}

func (b *Bot) handleDaily(c tele.Context) error {
	text := strings.TrimSpace(c.Message().Text)
	if text == "" {
		return nil
	}
	// Messages beginning with ? are RAG queries against the notes index.
	// Everything else is a daily note awaiting day-selection.
	if strings.HasPrefix(text, queryPrefix) {
		return b.handleQuery(c, strings.TrimSpace(strings.TrimPrefix(text, queryPrefix)))
	}
	return b.startPendingNote(c, text, c.Message(), "📝 File this note under:")
}

// handleQuery runs retrieval against the RAG sidecar, sends the question
// plus retrieved context to the answerer, and replies with the answer.
func (b *Bot) handleQuery(c tele.Context, question string) error {
	if b.rag == nil || b.answerer == nil {
		return c.Reply("🤔 RAG is not configured. Set RAG_SERVICE_URL.")
	}
	if question == "" {
		return c.Reply("Send `? <your question>` to search your notes.")
	}

	b.react(c.Message(), "👀")

	qctx, qcancel := context.WithTimeout(context.Background(), ragQueryTimeout)
	defer qcancel()

	hits, err := b.rag.Query(qctx, question, ragTopK)
	if err != nil {
		log.Printf("rag query: %v", err)
		b.react(c.Message(), "🤮")
		return c.Reply("🤮 Query failed.")
	}
	if len(hits) == 0 {
		b.react(c.Message(), "🤷")
		return c.Reply("No matching notes found.")
	}

	var contextBlock strings.Builder
	for _, h := range hits {
		fmt.Fprintf(&contextBlock, "[%s]\n%s\n\n", noteTitleFromKey(h.Key), h.Text)
	}

	actx, acancel := context.WithTimeout(context.Background(), groqTimeout)
	defer acancel()

	answer, err := b.answerer.Answer(actx, question, contextBlock.String())
	if err != nil {
		log.Printf("answer: %v", err)
		b.react(c.Message(), "🤮")
		return c.Reply("🤮 Answer failed.")
	}

	b.react(c.Message(), "👌")
	return c.Reply(answer)
}

// reindexInFlight guards against running two /reindex passes at once,
// which would hammer Ollama and produce confused progress messages.
var reindexInFlight sync.Mutex

// handleReindex walks the entire vault and indexes every markdown note.
// Notes whose content_hash already matches in the sidecar are skipped.
// Progress is reported by editing a single status message.
func (b *Bot) handleReindex(c tele.Context) error {
	if b.rag == nil || b.vault == nil || b.vault.Root() == "" {
		return c.Reply("🤔 RAG/vault is not configured. Set RAG_SERVICE_URL and VAULT_PATH.")
	}
	if !reindexInFlight.TryLock() {
		return c.Reply("⏳ A reindex is already running.")
	}

	status, err := c.Bot().Send(c.Chat(), "🔄 Reindexing vault...", &tele.SendOptions{ReplyTo: c.Message()})
	if err != nil {
		reindexInFlight.Unlock()
		return err
	}

	go func() {
		defer reindexInFlight.Unlock()

		var indexed, skipped, failed, total int
		lastEdit := time.Now()

		updateStatus := func(force bool) {
			if !force && time.Since(lastEdit) < 2*time.Second {
				return
			}
			lastEdit = time.Now()
			text := fmt.Sprintf("🔄 Reindexing... %d done (%d new, %d skipped, %d failed)", total, indexed, skipped, failed)
			if _, err := c.Bot().Edit(status, text); err != nil {
				log.Printf("reindex status edit: %v", err)
			}
		}

		walkErr := b.vault.Walk(func(e notes.VaultEntry) error {
			total++

			ctx, cancel := context.WithTimeout(context.Background(), ragIndexTimeout)
			res, err := b.rag.Index(ctx, e.Key, e.Content)
			cancel()

			switch {
			case err != nil:
				failed++
				log.Printf("reindex %s: %v", e.Key, err)
			case res.Skipped:
				skipped++
			default:
				indexed++
			}
			updateStatus(false)
			return nil
		})

		final := fmt.Sprintf("✅ Reindex done: %d total, %d new, %d skipped, %d failed", total, indexed, skipped, failed)
		if walkErr != nil {
			final = fmt.Sprintf("🤮 Reindex aborted at %d files: %v\n(%d new, %d skipped, %d failed)", total, walkErr, indexed, skipped, failed)
		}
		if _, err := c.Bot().Edit(status, final); err != nil {
			log.Printf("reindex final edit: %v", err)
		}
	}()
	return nil
}

// indexSavedNote fires the RAG indexer for a freshly saved daily note.
// The note's vault-relative path is computed from the store's PathFor
// and the configured vault root so the indexed key matches what
// /reindex would produce. Errors are logged only.
func (b *Bot) indexSavedNote(t time.Time, content string) {
	if b.rag == nil || b.vault == nil || b.vault.Root() == "" {
		return
	}
	abs := b.store.PathFor(t)
	key, err := b.vault.KeyFor(abs)
	if err != nil {
		log.Printf("rag index: cannot key %s: %v", abs, err)
		return
	}
	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), ragIndexTimeout)
		defer cancel()
		res, err := b.rag.Index(ctx, key, content)
		switch {
		case err != nil:
			log.Printf("rag index %s: %v", key, err)
		case res.Skipped:
			log.Printf("rag index %s: unchanged, skipped", key)
		default:
			log.Printf("rag indexed %s (%d chunks)", key, res.Chunks)
		}
	}()
}

// noteTitleFromKey turns a vault-relative path into a short label used
// in citations: drops the .md extension, keeps the parent folder for
// disambiguation.
func noteTitleFromKey(key string) string {
	return strings.TrimSuffix(key, ".md")
}

func (b *Bot) handleVoice(c tele.Context) error {
	voice := c.Message().Voice
	if voice == nil {
		return nil
	}

	reader, err := c.Bot().File(&voice.File)
	if err != nil {
		log.Printf("download voice: %v", err)
		b.react(c.Message(), "🤮")
		return nil
	}
	defer reader.Close()

	ctx, cancel := context.WithTimeout(context.Background(), transcribeTimeout)
	defer cancel()

	// Telegram voice notes are Opus in an Ogg container; ".ogg" is in
	// Whisper's accepted extension list.
	text, err := b.transcriber.Transcribe(ctx, reader, "voice.ogg")
	if err != nil {
		log.Printf("transcribe voice: %v", err)
		b.react(c.Message(), "🤮")
		return nil
	}

	text = strings.TrimSpace(text)
	if text == "" {
		b.react(c.Message(), "🤔")
		return nil
	}

	// Post the transcription as a standalone reply to the voice so it
	// stays visible alongside the voice after the day-selection prompt
	// is deleted. Status reactions land on this message, not the voice,
	// so the emoji sits next to readable text.
	transcript, err := c.Bot().Send(
		c.Chat(),
		"🎙️ "+text,
		&tele.SendOptions{ReplyTo: c.Message()},
	)
	if err != nil {
		log.Printf("send transcription: %v", err)
		b.react(c.Message(), "🤮")
		return nil
	}

	return b.startPendingNote(c, text, transcript, "📝 File this note under:")
}

// startPendingNote sends the day-selection prompt and stashes the note
// text against the prompt's message ID so the callback handler can find
// it when the user taps Yesterday or Today.
func (b *Bot) startPendingNote(c tele.Context, text string, reactTarget *tele.Message, promptMessage string) error {
	markup := &tele.ReplyMarkup{}
	markup.Inline(markup.Row(btnYesterday, btnToday))

	prompt, err := c.Bot().Send(
		c.Chat(),
		promptMessage,
		&tele.SendOptions{
			ReplyTo:     c.Message(),
			ReplyMarkup: markup,
		},
	)
	if err != nil {
		log.Printf("send day prompt: %v", err)
		b.react(reactTarget, "🤮")
		return nil
	}

	b.pendingMu.Lock()
	b.pending[prompt.ID] = &pendingNote{
		text:        text,
		reactTarget: reactTarget,
		createdAt:   time.Now(),
	}
	b.pendingMu.Unlock()

	return nil
}

// handleDayChoice returns a callback handler that files the pending note
// into the date offset by dayOffset days from "now" (0 = today, -1 = yesterday).
func (b *Bot) handleDayChoice(dayOffset int) tele.HandlerFunc {
	return func(c tele.Context) error {
		// Acknowledge the callback so Telegram clears the spinner on the
		// user's button. We don't care if this fails — it's UX polish.
		_ = c.Respond()

		cb := c.Callback()
		if cb == nil || cb.Message == nil {
			return nil
		}
		promptID := cb.Message.ID

		b.pendingMu.Lock()
		pending, ok := b.pending[promptID]
		if ok {
			delete(b.pending, promptID)
		}
		b.pendingMu.Unlock()

		if !ok {
			_ = c.Edit("⚠️ This note expired. Send it again.", &tele.ReplyMarkup{})
			return nil
		}

		target := time.Now().UTC().AddDate(0, 0, dayOffset)
		dateLabel := target.Format("2006-01-02")

		existing, err := b.store.Load(target)
		if err != nil {
			log.Printf("load note: %v", err)
			_ = c.Edit("🤮 Failed to load "+dateLabel, &tele.ReplyMarkup{})
			b.react(pending.reactTarget, "🤮")
			return nil
		}

		merged := notes.AppendEntry(existing, pending.text)

		ctx, cancel := context.WithTimeout(context.Background(), groqTimeout)
		defer cancel()

		refined, err := b.refiner.Refine(ctx, merged)
		if err != nil {
			log.Printf("groq refine: %v", err)
			_ = c.Edit("🤮 Refine failed for "+dateLabel, &tele.ReplyMarkup{})
			b.react(pending.reactTarget, "🤮")
			return nil
		}

		if err := b.store.Save(target, refined); err != nil {
			log.Printf("save note: %v", err)
			_ = c.Edit("🤮 Save failed for "+dateLabel, &tele.ReplyMarkup{})
			b.react(pending.reactTarget, "🤮")
			return nil
		}

		b.indexSavedNote(target, refined)

		_ = c.Delete()
		b.react(pending.reactTarget, "👌")
		return nil
	}
}

// cleanupLoop periodically evicts pending notes whose prompt the user
// never answered, so the map doesn't grow without bound.
func (b *Bot) cleanupLoop() {
	ticker := time.NewTicker(cleanupInterval)
	defer ticker.Stop()
	for range ticker.C {
		cutoff := time.Now().Add(-pendingTTL)
		b.pendingMu.Lock()
		for id, note := range b.pending {
			if note.createdAt.Before(cutoff) {
				delete(b.pending, id)
			}
		}
		b.pendingMu.Unlock()
	}
}

// react sends an emoji reaction on the given message using the raw Bot API.
func (b *Bot) react(msg *tele.Message, emoji string) {
	if msg == nil {
		return
	}
	params := map[string]interface{}{
		"chat_id":    msg.Chat.ID,
		"message_id": msg.ID,
		"reaction":   []map[string]string{{"type": "emoji", "emoji": emoji}},
	}
	if _, err := b.tele.Raw("setMessageReaction", params); err != nil {
		log.Printf("react: %v", err)
	}
}

// preview returns the first n characters of s, appending an ellipsis if
// the string was truncated. It is rune-safe.
func preview(s string, n int) string {
	runes := []rune(s)
	if len(runes) <= n {
		return s
	}
	return string(runes[:n]) + "..."
}
