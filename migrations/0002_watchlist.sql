-- Watchlist and new-filing feed for the dashboard. A filing is recorded once (accession is unique).
CREATE TABLE IF NOT EXISTS watchlist (
    ticker        text PRIMARY KEY,
    cik           text NOT NULL,
    name          text,
    forms         text[] NOT NULL DEFAULT ARRAY['10-K', '10-Q', '8-K'],
    auto_analyze  boolean NOT NULL DEFAULT false,
    added_at      timestamptz NOT NULL DEFAULT now(),
    last_checked_at timestamptz,
    last_error    text
);

CREATE TABLE IF NOT EXISTS filings_seen (
    accession         text PRIMARY KEY,
    ticker            text NOT NULL REFERENCES watchlist (ticker) ON DELETE CASCADE,
    cik               text NOT NULL,
    form              text NOT NULL,
    filing_date       date NOT NULL,
    report_date       date,
    primary_document  text NOT NULL,
    url               text NOT NULL,
    first_seen_at     timestamptz NOT NULL DEFAULT now(),
    baseline          boolean NOT NULL DEFAULT false,   -- already on file when the ticker was added
    job_ids           jsonb NOT NULL DEFAULT '[]'::jsonb
);
CREATE INDEX IF NOT EXISTS filings_seen_recent ON filings_seen (filing_date DESC, first_seen_at DESC);
