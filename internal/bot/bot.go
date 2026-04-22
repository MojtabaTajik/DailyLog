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
	tele "gopkg.in/telebot.v3"
)

const (
	previewLimit      = 200
	groqTimeout       = 90 * time.Second
	transcribeTimeout = 120 * time.Second
	pollingTimeout    = 10 * time.Second
	pendingTTL        = 30 * time.Minute
	cleanupInterval   = 5 * time.Minute
)

// Inline buttons attached to each note prompt. The Unique field is what
// telebot uses to route the callback back to the correct handler.
var (
	btnYesterday = tele.Btn{Unique: "note_yesterday", Text: "Yesterday"}
	btnToday     = tele.Btn{Unique: "note_today", Text: "Today"}
)

// Refiner is the subset of the Groq client used by the bot. Defining it
// here keeps the bot decoupled from the concrete transport and makes it
// trivial to swap or stub in tests.
type Refiner interface {
	Refine(ctx context.Context, raw string) (string, error)
}

// Transcriber converts an audio stream to text. filename must carry a
// Whisper-recognized extension (e.g. ".ogg" for Telegram voice notes).
type Transcriber interface {
	Transcribe(ctx context.Context, audio io.Reader, filename string) (string, error)
}

// NoteStore is the persistence contract the bot depends on.
type NoteStore interface {
	Load(t time.Time) (string, error)
	Save(t time.Time, content string) error
}

// pendingNote holds a note awaiting the user's day-selection click.
type pendingNote struct {
	text        string
	userMessage *tele.Message
	createdAt   time.Time
}

// Bot wires together Telegram, the note store, and the AI refiner.
type Bot struct {
	cfg         *config.Config
	tele        *tele.Bot
	store       NoteStore
	refiner     Refiner
	transcriber Transcriber

	pendingMu sync.Mutex
	pending   map[int]*pendingNote
}

// New constructs a Bot and registers handlers. It returns an error if
// the underlying Telegram client cannot be initialized.
func New(cfg *config.Config, store NoteStore, refiner Refiner, transcriber Transcriber) (*Bot, error) {
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
		refiner:     refiner,
		transcriber: transcriber,
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
		"/help — show this message",
	}, "\n"))
}

func (b *Bot) handleDaily(c tele.Context) error {
	// Every non‑command text message is treated as a daily note awaiting
	// a day-selection from the user.
	text := strings.TrimSpace(c.Message().Text)
	if text == "" {
		return nil
	}
	return b.startPendingNote(c, text, "📝 File this note under:")
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

	prompt := fmt.Sprintf("🎙️ Transcribed:\n%s\n\n📝 File this note under:", text)
	return b.startPendingNote(c, text, prompt)
}

// startPendingNote sends the day-selection prompt and stashes the note
// text against the prompt's message ID so the callback handler can find
// it when the user taps Yesterday or Today.
func (b *Bot) startPendingNote(c tele.Context, text, promptMessage string) error {
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
		b.react(c.Message(), "🤮")
		return nil
	}

	b.pendingMu.Lock()
	b.pending[prompt.ID] = &pendingNote{
		text:        text,
		userMessage: c.Message(),
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
			b.react(pending.userMessage, "🤮")
			return nil
		}

		merged := notes.AppendEntry(existing, pending.text)

		ctx, cancel := context.WithTimeout(context.Background(), groqTimeout)
		defer cancel()

		refined, err := b.refiner.Refine(ctx, merged)
		if err != nil {
			log.Printf("groq refine: %v", err)
			_ = c.Edit("🤮 Refine failed for "+dateLabel, &tele.ReplyMarkup{})
			b.react(pending.userMessage, "🤮")
			return nil
		}

		if err := b.store.Save(target, refined); err != nil {
			log.Printf("save note: %v", err)
			_ = c.Edit("🤮 Save failed for "+dateLabel, &tele.ReplyMarkup{})
			b.react(pending.userMessage, "🤮")
			return nil
		}

		_ = c.Delete()
		b.react(pending.userMessage, "👌")
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
