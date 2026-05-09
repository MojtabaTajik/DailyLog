package bot

import (
	"testing"
	"time"
)

func TestParseTemporalRange(t *testing.T) {
	today := time.Date(2026, 5, 9, 12, 0, 0, 0, time.UTC) // Saturday

	cases := []struct {
		q      string
		want   string // "" means no match expected
		wantTo string
	}{
		{"what did I do today?", "2026-05-09", "2026-05-09"},
		{"yesterday", "2026-05-08", "2026-05-08"},
		{"summarize this week", "2026-05-04", "2026-05-10"},
		{"what happened last week", "2026-04-27", "2026-05-03"},
		{"what happened this month", "2026-05-01", "2026-05-31"},
		{"what happened last month", "2026-04-01", "2026-04-30"},
		{"on 2025-05-08 what did I do", "2025-05-08", "2025-05-08"},
		{"what did I do last thursday", "2026-05-07", "2026-05-07"},
		{"what did I do on thursday", "2026-05-07", "2026-05-07"},
		{"vulwall auth refactor status", "", ""},
	}

	for _, c := range cases {
		from, to, ok := parseTemporalRange(c.q, today)
		if c.want == "" {
			if ok {
				t.Errorf("%q: expected no match, got %s..%s", c.q, from.Format("2006-01-02"), to.Format("2006-01-02"))
			}
			continue
		}
		if !ok {
			t.Errorf("%q: expected match, got none", c.q)
			continue
		}
		gotFrom := from.Format("2006-01-02")
		gotTo := to.Format("2006-01-02")
		if gotFrom != c.want || gotTo != c.wantTo {
			t.Errorf("%q: got %s..%s, want %s..%s", c.q, gotFrom, gotTo, c.want, c.wantTo)
		}
	}
}

func TestStartOfWeek(t *testing.T) {
	// 2026-05-09 is a Saturday; Monday-based start should be 2026-05-04.
	sat := time.Date(2026, 5, 9, 12, 0, 0, 0, time.UTC)
	got := startOfWeek(sat)
	if got.Format("2006-01-02") != "2026-05-04" {
		t.Errorf("startOfWeek(sat) = %s, want 2026-05-04", got.Format("2006-01-02"))
	}

	// Sunday should map to the *previous* Monday.
	sun := time.Date(2026, 5, 10, 12, 0, 0, 0, time.UTC)
	got = startOfWeek(sun)
	if got.Format("2006-01-02") != "2026-05-04" {
		t.Errorf("startOfWeek(sun) = %s, want 2026-05-04", got.Format("2006-01-02"))
	}

	// Monday should map to itself.
	mon := time.Date(2026, 5, 4, 12, 0, 0, 0, time.UTC)
	got = startOfWeek(mon)
	if got.Format("2006-01-02") != "2026-05-04" {
		t.Errorf("startOfWeek(mon) = %s, want 2026-05-04", got.Format("2006-01-02"))
	}
}
