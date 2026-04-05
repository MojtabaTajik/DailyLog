package main

import (
	"log"

	"github.com/mojix/dailylog/internal/bot"
	"github.com/mojix/dailylog/internal/config"
	"github.com/mojix/dailylog/internal/groq"
	"github.com/mojix/dailylog/internal/notes"
)

func main() {
	cfg, err := config.Load()
	if err != nil {
		log.Fatalf("config: %v", err)
	}

	store := notes.NewStore(cfg.DailyNotesPath)
	groqClient := groq.NewClient(cfg.GroqAPIKey, cfg.GroqModel, cfg.GroqSystemPrompt)

	b, err := bot.New(cfg, store, groqClient)
	if err != nil {
		log.Fatalf("bot init: %v", err)
	}

	log.Println("dailylog bot starting...")
	b.Start()
}
