package bot

import (
	"context"
	"fmt"
	"log"
	"strings"
	"time"

	"github.com/mojix/dailylog/internal/config"
	"github.com/mojix/dailylog/internal/notes"
	tele "gopkg.in/telebot.v3"
)

const (
	previewLimit   = 200
	groqTimeout    = 90 * time.Second
	pollingTimeout = 10 * time.Second
)

// Refiner is the subset of the Groq client used by the bot. Defining it
// here keeps the bot decoupled from the concrete transport and makes it
// trivial to swap or stub in tests.
type Refiner interface {
	Refine(ctx context.Context, raw string) (string, error)
}

// NoteStore is the persistence contract the bot depends on.
type NoteStore interface {
	Load(t time.Time) (string, error)
	Save(t time.Time, content string) error
}

// Bot wires together Telegram, the note store, and the AI refiner.
type Bot struct {
	cfg     *config.Config
	tele    *tele.Bot
	store   NoteStore
	refiner Refiner
}

// New constructs a Bot and registers handlers. It returns an error if
// the underlying Telegram client cannot be initialized.
func New(cfg *config.Config, store NoteStore, refiner Refiner) (*Bot, error) {
	settings := tele.Settings{
		Token:  cfg.TelegramToken,
		Poller: &tele.LongPoller{Timeout: pollingTimeout},
	}

	tb, err := tele.NewBot(settings)
	if err != nil {
		return nil, fmt.Errorf("telebot init: %w", err)
	}

	b := &Bot{cfg: cfg, tele: tb, store: store, refiner: refiner}
	b.registerHandlers()
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
		"Send any text message and it will be appended to today's log and refined into the fixed section structure.",
		"/help — show this message",
	}, "\n"))
}

func (b *Bot) handleDaily(c tele.Context) error {
	// Every non‑command text message is treated as a daily note.
	text := strings.TrimSpace(c.Message().Text)
	if text == "" {
		return nil
	}

	now := time.Now().UTC()

	existing, err := b.store.Load(now)
	if err != nil {
		log.Printf("load note: %v", err)
		b.react(c, "🤮")
		return nil
	}

	merged := notes.AppendEntry(existing, text)

	ctx, cancel := context.WithTimeout(context.Background(), groqTimeout)
	defer cancel()

	refined, err := b.refiner.Refine(ctx, merged)
	if err != nil {
		log.Printf("groq refine: %v", err)
		b.react(c, "🤮")
		return nil
	}

	if err := b.store.Save(now, refined); err != nil {
		log.Printf("save note: %v", err)
		b.react(c, "🤮")
		return nil
	}

	b.react(c, "👍")
	return nil
}

// react sends an emoji reaction on the message using the raw Bot API.
func (b *Bot) react(c tele.Context, emoji string) {
	params := map[string]interface{}{
		"chat_id":    c.Chat().ID,
		"message_id": c.Message().ID,
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
