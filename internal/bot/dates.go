package bot

import (
	"regexp"
	"strings"
	"time"
)

var (
	isoDateRe = regexp.MustCompile(`\b(\d{4}-\d{2}-\d{2})\b`)
	yearMonRe = regexp.MustCompile(`\b(\d{4}-\d{2})\b`)
	weekdayRe = regexp.MustCompile(`\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b`)
	lastWdRe  = regexp.MustCompile(`\blast\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b`)
)

var weekdayIndex = map[string]time.Weekday{
	"sunday":    time.Sunday,
	"monday":    time.Monday,
	"tuesday":   time.Tuesday,
	"wednesday": time.Wednesday,
	"thursday":  time.Thursday,
	"friday":    time.Friday,
	"saturday":  time.Saturday,
}

// parseTemporalRange extracts an inclusive date range from a question
// using simple keyword matching. It mirrors the Python parser in the
// rag-service so the Go side can decide locally whether to bypass RAG
// for small ranges. Week starts Monday.
func parseTemporalRange(q string, today time.Time) (from, to time.Time, ok bool) {
	if q == "" {
		return time.Time{}, time.Time{}, false
	}
	today = atMidnight(today)
	s := strings.ToLower(q)

	if m := isoDateRe.FindStringSubmatch(s); m != nil {
		d, err := time.ParseInLocation("2006-01-02", m[1], today.Location())
		if err == nil {
			return d, d, true
		}
	}

	// Year-month: only match if not part of a full ISO date already handled above.
	// The yearMonRe will also match the YYYY-MM prefix of YYYY-MM-DD, so
	// we deliberately attempt this only after the ISO branch failed.
	if m := yearMonRe.FindStringSubmatch(s); m != nil && !strings.Contains(s, m[1]+"-") {
		first, err := time.ParseInLocation("2006-01-02", m[1]+"-01", today.Location())
		if err == nil {
			next := first.AddDate(0, 1, 0)
			last := next.AddDate(0, 0, -1)
			return first, last, true
		}
	}

	if strings.Contains(s, "today") {
		return today, today, true
	}
	if strings.Contains(s, "yesterday") {
		d := today.AddDate(0, 0, -1)
		return d, d, true
	}
	if strings.Contains(s, "this week") {
		start := startOfWeek(today)
		return start, start.AddDate(0, 0, 6), true
	}
	if strings.Contains(s, "last week") {
		start := startOfWeek(today).AddDate(0, 0, -7)
		return start, start.AddDate(0, 0, 6), true
	}
	if strings.Contains(s, "this month") {
		start := time.Date(today.Year(), today.Month(), 1, 0, 0, 0, 0, today.Location())
		end := start.AddDate(0, 1, -1)
		return start, end, true
	}
	if strings.Contains(s, "last month") {
		firstThis := time.Date(today.Year(), today.Month(), 1, 0, 0, 0, 0, today.Location())
		end := firstThis.AddDate(0, 0, -1)
		start := time.Date(end.Year(), end.Month(), 1, 0, 0, 0, 0, today.Location())
		return start, end, true
	}

	if m := lastWdRe.FindStringSubmatch(s); m != nil {
		wd := weekdayIndex[m[1]]
		delta := (int(today.Weekday()) - int(wd) + 7) % 7
		if delta == 0 {
			delta = 7
		}
		d := today.AddDate(0, 0, -delta)
		return d, d, true
	}
	if m := weekdayRe.FindStringSubmatch(s); m != nil {
		wd := weekdayIndex[m[1]]
		delta := (int(today.Weekday()) - int(wd) + 7) % 7
		d := today.AddDate(0, 0, -delta)
		return d, d, true
	}

	return time.Time{}, time.Time{}, false
}

func atMidnight(t time.Time) time.Time {
	return time.Date(t.Year(), t.Month(), t.Day(), 0, 0, 0, 0, t.Location())
}

func startOfWeek(t time.Time) time.Time {
	// Go: Sunday=0, Monday=1, ..., Saturday=6. We want Monday-based week.
	wd := int(t.Weekday())
	if wd == 0 {
		wd = 7
	}
	return t.AddDate(0, 0, -(wd - 1))
}
