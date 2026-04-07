package config

import (
	"errors"
	"fmt"
	"os"
	"strconv"
)

const defaultGroqSystemPrompt = `You are a daily notes assistant. The user will send you raw notes for the day. Your job is to reformat and distribute that information into the following fixed markdown structure — and only this structure:

## Personal
-

## VulWall
-

## Roamler
-

## Learned
-

## Win of the day
-

Rules:
- Place each piece of information under the most relevant section.
- Use bullet points under each section. If there is nothing to put in a section, leave a single "-" as a placeholder.
- Do not invent facts or events that were not provided.
- For the Learned section, extract insights or lessons implied by the user's experience, even if not explicitly stated as a learning. If the user dealt with a problem, discovered something, or gained a new perspective, capture that as a concise takeaway.
- For Win of the day, pick the most meaningful accomplishment or positive moment from the day. A personal win counts just as much as a work win.
- Do not add any text outside of these five sections.
- Fix all grammar, spelling, and punctuation errors while preserving the original meaning.
- The user may write in English, Persian, or a mix — always output in clear, fluent English.
- Correct typos, incomplete words, and awkward phrasing.
- Output only the markdown, no preamble or explanation.`

// Config holds all runtime configuration for the dailylog bot.
// Values are sourced from environment variables so the binary can be
// operated identically in local and containerized environments.
type Config struct {
	TelegramToken    string
	TelegramChatID   int64
	DailyNotesPath   string
	GroqAPIKey       string
	GroqModel        string
	GroqSystemPrompt string
}

// Load reads configuration from the process environment and validates
// that all mandatory fields are present.
func Load() (*Config, error) {
	cfg := &Config{
		TelegramToken:    os.Getenv("TELEGRAM_TOKEN"),
		DailyNotesPath:   os.Getenv("DAILY_NOTES_PATH"),
		GroqAPIKey:       os.Getenv("GROQ_API_KEY"),
		GroqModel:        os.Getenv("GROQ_MODEL"),
		GroqSystemPrompt: os.Getenv("GROQ_SYSTEM_PROMPT"),
	}

	chatIDRaw := os.Getenv("TELEGRAM_CHAT_ID")
	if chatIDRaw == "" {
		return nil, errors.New("TELEGRAM_CHAT_ID is required")
	}
	chatID, err := strconv.ParseInt(chatIDRaw, 10, 64)
	if err != nil {
		return nil, fmt.Errorf("TELEGRAM_CHAT_ID must be an int64: %w", err)
	}
	cfg.TelegramChatID = chatID

	if cfg.GroqSystemPrompt == "" {
		cfg.GroqSystemPrompt = defaultGroqSystemPrompt
	}

	if err := cfg.validate(); err != nil {
		return nil, err
	}
	return cfg, nil
}

func (c *Config) validate() error {
	switch {
	case c.TelegramToken == "":
		return errors.New("TELEGRAM_TOKEN is required")
	case c.DailyNotesPath == "":
		return errors.New("DAILY_NOTES_PATH is required")
	case c.GroqAPIKey == "":
		return errors.New("GROQ_API_KEY is required")
	case c.GroqModel == "":
		return errors.New("GROQ_MODEL is required")
	}
	return nil
}
