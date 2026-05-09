package main

import (
	"log"

	"github.com/mojix/dailylog/internal/bot"
	"github.com/mojix/dailylog/internal/config"
	"github.com/mojix/dailylog/internal/groq"
	"github.com/mojix/dailylog/internal/notes"
	"github.com/mojix/dailylog/internal/rag"
)

func main() {
	cfg, err := config.Load()
	if err != nil {
		log.Fatalf("config: %v", err)
	}

	store := notes.NewStore(cfg.DailyNotesPath)
	vault := notes.NewVaultStore(cfg.VaultPath)
	groqClient := groq.NewClient(cfg.GroqAPIKey, cfg.GroqModel, cfg.GroqTranscribeModel, cfg.GroqSystemPrompt)

	var ragClient bot.RagClient
	if cfg.RagServiceURL != "" {
		ragClient = rag.NewClient(cfg.RagServiceURL)
		log.Printf("rag enabled: %s (vault=%s)", cfg.RagServiceURL, cfg.VaultPath)
	} else {
		log.Println("rag disabled (RAG_SERVICE_URL not set)")
	}

	b, err := bot.New(cfg, store, vault, groqClient, groqClient, groqClient, groqClient, ragClient)
	if err != nil {
		log.Fatalf("bot init: %v", err)
	}

	log.Println("dailylog bot starting...")
	b.Start()
}
