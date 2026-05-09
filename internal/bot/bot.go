package bot

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
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
	groqTimeout       = 90 * time.Second
	transcribeTimeout = 120 * time.Second
	pollingTimeout    = 10 * time.Second
	pendingTTL        = 30 * time.Minute
	cleanupInterval   = 5 * time.Minute
	ragIndexTimeout   = 60 * time.Second
	ragQueryTimeout   = 30 * time.Second
	rewriteTimeout    = 15 * time.Second
	ragTopK           = 6
	queryPrefix       = "?"
	maxBypassDays     = 14
	maxRewriteRunes   = 200
	telegramReplyMax  = 4000
	ftsDebounce       = 5 * time.Second
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

// Rewriter expands a terse user question into 1-3 search queries plus
// an optional date range. An empty queries slice or empty dates means
// "no useful expansion"; the caller should fall back to the raw input.
type Rewriter interface {
	Rewrite(ctx context.Context, question string) (queries []string, dateFrom, dateTo string, err error)
}

// RagClient is the subset of the RAG sidecar used by the bot.
type RagClient interface {
	Index(ctx context.Context, key, content string, rebuildFTS bool) (rag.IndexResult, error)
	Query(ctx context.Context, q string, k int, dateFrom, dateTo string) ([]rag.Hit, error)
	Status(ctx context.Context) (map[string]string, error)
	RebuildFTS(ctx context.Context) error
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
// reactTarget is the message that receives status reactions: for voice
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
	rewriter    Rewriter // may be nil to disable query rewriting
	transcriber Transcriber
	rag         RagClient // may be nil if RAG is not configured

	pendingMu sync.Mutex
	pending   map[int]*pendingNote

	// FTS rebuild debouncing: many sequential single-note saves produce
	// one rebuild instead of N.
	ftsMu    sync.Mutex
	ftsTimer *time.Timer
}

// New constructs a Bot and registers handlers. It returns an error if
// the underlying Telegram client cannot be initialized. ragClient and
// rewriter may both be nil to disable indexing/expansion.
func New(
	cfg *config.Config,
	store NoteStore,
	vault Vault,
	refiner Refiner,
	answerer Answerer,
	rewriter Rewriter,
	transcriber Transcriber,
	ragClient RagClient,
) (*Bot, error) {
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
		rewriter:    rewriter,
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
	if strings.HasPrefix(text, queryPrefix) {
		return b.handleQuery(c, strings.TrimSpace(strings.TrimPrefix(text, queryPrefix)))
	}
	return b.startPendingNote(c, text, c.Message(), "📝 File this note under:")
}

// handleQuery answers a `?` question. Three paths, in order:
//  1. If the question references a small date range (≤ maxBypassDays),
//     load those daily notes directly and skip retrieval entirely.
//  2. Otherwise, optionally ask the rewriter to expand the question
//     into a search-friendly form plus a possible ISO date range.
//  3. Run the rag sidecar and answer from its hits.
func (b *Bot) handleQuery(c tele.Context, question string) error {
	if b.rag == nil || b.answerer == nil {
		return c.Reply("🤔 RAG is not configured. Set RAG_SERVICE_URL.")
	}
	if question == "" {
		return c.Reply("Send `? <your question>` to search your notes.")
	}

	b.react(c.Message(), "👀")

	now := time.Now().Local()
	parseFrom, parseTo, parseOK := parseTemporalRange(question, now)

	if parseOK {
		days := int(parseTo.Sub(parseFrom).Hours()/24) + 1
		if days >= 1 && days <= maxBypassDays {
			return b.answerFromRange(c, question, parseFrom, parseTo)
		}
	}

	rewritten := question
	var rewriteFrom, rewriteTo string
	if b.rewriter != nil && len([]rune(question)) <= maxRewriteRunes {
		rwctx, rwcancel := context.WithTimeout(context.Background(), rewriteTimeout)
		queries, df, dt, err := b.rewriter.Rewrite(rwctx, question)
		rwcancel()
		if err != nil {
			log.Printf("rewrite: %v (using raw question)", err)
		} else {
			if len(queries) > 0 && strings.TrimSpace(queries[0]) != "" {
				rewritten = strings.TrimSpace(queries[0])
			}
			rewriteFrom, rewriteTo = strings.TrimSpace(df), strings.TrimSpace(dt)
		}
	}

	dateFrom, dateTo := rewriteFrom, rewriteTo
	if dateFrom == "" && dateTo == "" && parseOK {
		dateFrom = parseFrom.Format("2006-01-02")
		dateTo = parseTo.Format("2006-01-02")
	}

	qctx, qcancel := context.WithTimeout(context.Background(), ragQueryTimeout)
	defer qcancel()

	hits, err := b.rag.Query(qctx, rewritten, ragTopK, dateFrom, dateTo)
	if err != nil {
		log.Printf("rag query: %v", err)
		b.react(c.Message(), "🤮")
		return c.Reply("🤮 Query failed.")
	}
	if len(hits) == 0 {
		b.react(c.Message(), "🤷")
		return c.Reply("No matching notes found.")
	}

	// Hits already carry "Date: ..." / "Section: ..." prefixes from the
	// sidecar, so we just glue them together with separators.
	var contextBlock strings.Builder
	for i, h := range hits {
		if i > 0 {
			contextBlock.WriteString("\n\n---\n\n")
		}
		contextBlock.WriteString(h.Text)
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
	return b.replyLong(c, answer)
}

// answerFromRange loads the daily notes for [from, to] inclusive and
// hands them to the answerer verbatim. Used as a bypass when the user's
// question references a small date range; faster, cheaper, and gives
// the model the entire note rather than a few retrieved chunks.
func (b *Bot) answerFromRange(c tele.Context, question string, from, to time.Time) error {
	var ctxBlock strings.Builder
	found := 0
	for d := from; !d.After(to); d = d.AddDate(0, 0, 1) {
		body, err := b.store.Load(d)
		if err != nil {
			log.Printf("range load %s: %v", d.Format("2006-01-02"), err)
			continue
		}
		body = strings.TrimSpace(body)
		if body == "" {
			continue
		}
		if found > 0 {
			ctxBlock.WriteString("\n\n---\n\n")
		}
		fmt.Fprintf(&ctxBlock, "Date: %s (%s)\n%s",
			d.Format("2006-01-02"), d.Format("Mon"), body)
		found++
	}

	if found == 0 {
		b.react(c.Message(), "🤷")
		return c.Reply("No notes found in that range.")
	}

	actx, acancel := context.WithTimeout(context.Background(), groqTimeout)
	defer acancel()

	answer, err := b.answerer.Answer(actx, question, ctxBlock.String())
	if err != nil {
		log.Printf("answer (range): %v", err)
		b.react(c.Message(), "🤮")
		return c.Reply("🤮 Answer failed.")
	}

	b.react(c.Message(), "👌")
	return b.replyLong(c, answer)
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

		statusCtx, cancelStatus := context.WithTimeout(context.Background(), ragQueryTimeout)
		known, err := b.rag.Status(statusCtx)
		cancelStatus()
		if err != nil {
			log.Printf("reindex status fetch: %v", err)
			known = map[string]string{}
		}

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

			h := sha256.Sum256([]byte(e.Content))
			localHash := hex.EncodeToString(h[:])
			if known[e.Key] == localHash {
				skipped++
				updateStatus(false)
				return nil
			}

			ctx, cancel := context.WithTimeout(context.Background(), ragIndexTimeout)
			res, err := b.rag.Index(ctx, e.Key, e.Content, false)
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

		if indexed > 0 {
			fctx, fcancel := context.WithTimeout(context.Background(), ragIndexTimeout)
			if err := b.rag.RebuildFTS(fctx); err != nil {
				log.Printf("reindex rebuild-fts: %v", err)
			}
			fcancel()
		}

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

// indexSavedNote fires the RAG indexer for a freshly saved daily note,
// then schedules a debounced FTS rebuild. Errors are logged only.
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
		// Rebuild_fts=false: a debounced background rebuild handles
		// bursts of saves with one FTS pass instead of N.
		res, err := b.rag.Index(ctx, key, content, false)
		switch {
		case err != nil:
			log.Printf("rag index %s: %v", key, err)
			return
		case res.Skipped:
			log.Printf("rag index %s: unchanged, skipped", key)
			return
		default:
			log.Printf("rag indexed %s (%d chunks)", key, res.Chunks)
		}
		b.scheduleFTSRebuild()
	}()
}

// scheduleFTSRebuild fires a single FTS rebuild ftsDebounce after the
// most recent call. Subsequent calls within the window reset the timer.
func (b *Bot) scheduleFTSRebuild() {
	b.ftsMu.Lock()
	defer b.ftsMu.Unlock()
	if b.ftsTimer != nil {
		b.ftsTimer.Stop()
	}
	b.ftsTimer = time.AfterFunc(ftsDebounce, func() {
		ctx, cancel := context.WithTimeout(context.Background(), ragIndexTimeout)
		defer cancel()
		if err := b.rag.RebuildFTS(ctx); err != nil {
			log.Printf("debounced fts rebuild: %v", err)
			return
		}
		log.Printf("fts index rebuilt")
	})
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

		// Local time, not UTC: near-midnight messages must file under the
		// user's local day, otherwise a 23:55 note ends up under tomorrow.
		target := time.Now().Local().AddDate(0, 0, dayOffset)
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

// replyLong sends text as a normal reply when it fits Telegram's 4096
// character limit; otherwise it sends the first chunk as text and the
// remainder as a `.txt` document attachment.
func (b *Bot) replyLong(c tele.Context, text string) error {
	runes := []rune(text)
	if len(runes) <= telegramReplyMax {
		return c.Reply(text)
	}
	head := string(runes[:telegramReplyMax])
	tail := string(runes[telegramReplyMax:])
	if err := c.Reply(head); err != nil {
		return err
	}
	doc := &tele.Document{
		File:     tele.FromReader(bytes.NewReader([]byte(tail))),
		FileName: "answer-overflow.txt",
	}
	return c.Reply(doc)
}
