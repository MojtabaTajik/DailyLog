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

	b.tele.Handle("/log", b.handleDaily)
	b.tele.Handle("/help", b.handleHelp)
	b.tele.Handle("/start", b.handleHelp)
	b.tele.Handle(tele.OnText, b.handleFallback)
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
		"dailylog commands:",
		"/log <note> — append a note to today's log and refine it",
		"/help — show this message",
	}, "\n"))
}

func (b *Bot) handleFallback(c tele.Context) error {
	return c.Send("Use /log <note> to log your daily note.")
}

func (b *Bot) handleDaily(c tele.Context) error {
	text := strings.TrimSpace(c.Message().Payload)
	if text == "" {
		return c.Send("Usage: /log <your note>")
	}

	now := time.Now().UTC()

	existing, err := b.store.Load(now)
	if err != nil {
		log.Printf("load note: %v", err)
		b.react(c, "⚠")
		return nil
	}

	merged := notes.AppendEntry(existing, text, now)

	ctx, cancel := context.WithTimeout(context.Background(), groqTimeout)
	defer cancel()

	refined, err := b.refiner.Refine(ctx, merged)
	if err != nil {
		log.Printf("groq refine: %v", err)
		b.react(c, "⚠")
		return nil
	}

	if err := b.store.Save(now, refined); err != nil {
		log.Printf("save note: %v", err)
		b.react(c, "⚠")
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
