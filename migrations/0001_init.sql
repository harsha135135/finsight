-- FinSight schema v1. Application records; LangGraph checkpoint tables are created by PostgresSaver.setup().

CREATE TABLE research_jobs (
    id                  uuid PRIMARY KEY,
    request             jsonb NOT NULL,                 -- company, question, periods, metrics, cutoff date
    model_config        jsonb NOT NULL,                 -- immutable provider/model/settings snapshot at admission
    status              text NOT NULL DEFAULT 'queued'
                        CHECK (status IN ('queued', 'running', 'completed', 'insufficient_evidence',
                                          'blocked', 'failed', 'cancelled')),
    stage               text,
    outcome_reason      text,
    cancel_requested    boolean NOT NULL DEFAULT false,
    owner_token         uuid,                           -- ownership of the current worker; checked on every write
    owner_worker        text,
    owner_backend_pid   integer,                        -- the worker's dedicated session (lock + checkpointer)
    heartbeat_backend_pid integer,
    heartbeat_at        timestamptz,
    repairs             jsonb NOT NULL DEFAULT '{}',    -- per failure category, at most one each, two in total
    idempotency_key     text UNIQUE,
    created_at          timestamptz NOT NULL DEFAULT now(),
    updated_at          timestamptz NOT NULL DEFAULT now(),
    finished_at         timestamptz
);
CREATE INDEX research_jobs_queued ON research_jobs (created_at) WHERE status = 'queued';
CREATE INDEX research_jobs_running ON research_jobs (heartbeat_at) WHERE status = 'running';

CREATE TABLE job_events (
    id          bigserial PRIMARY KEY,
    job_id      uuid NOT NULL REFERENCES research_jobs (id),
    at          timestamptz NOT NULL DEFAULT now(),
    kind        text NOT NULL,
    stage       text,
    detail      jsonb NOT NULL DEFAULT '{}'
);
CREATE INDEX job_events_job ON job_events (job_id, id);

CREATE TABLE documents (
    id              text PRIMARY KEY,
    source_uri      text NOT NULL,
    sha256          text NOT NULL,
    artifact_path   text NOT NULL,
    accession       text,
    cik             text,
    company         text,
    form            text,
    fiscal_year     integer,
    period_end      date,
    filing_date     date,
    content_type    text,
    size_bytes      bigint,
    retrieved_at    timestamptz NOT NULL,
    parser_version  text,
    UNIQUE (source_uri, sha256)
);

-- Aggregated API responses (e.g. SEC companyfacts) change over time: stored as timestamped snapshots.
CREATE TABLE source_snapshots (
    id              bigserial PRIMARY KEY,
    kind            text NOT NULL,
    source_uri      text NOT NULL,
    sha256          text NOT NULL,
    artifact_path   text NOT NULL,
    retrieved_at    timestamptz NOT NULL
);

CREATE TABLE passages (
    id              text PRIMARY KEY,
    document_id     text NOT NULL REFERENCES documents (id),
    section         text NOT NULL,
    location        jsonb NOT NULL DEFAULT '{}',
    start_offset    integer NOT NULL,
    end_offset      integer NOT NULL,
    text            text NOT NULL,
    tsv             tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
);
CREATE INDEX passages_tsv ON passages USING gin (tsv);
CREATE INDEX passages_document ON passages (document_id);

CREATE TABLE job_documents (
    job_id      uuid NOT NULL REFERENCES research_jobs (id),
    document_id text NOT NULL REFERENCES documents (id),
    PRIMARY KEY (job_id, document_id)
);

-- Every model call, paid or not, from FinSight jobs or ModelForge runs. Also the authoritative spend ledger:
-- committed spend = completed cost + reservations of reserved/sent/unknown attempts.
CREATE TABLE model_attempts (
    id              uuid PRIMARY KEY,
    scope           text NOT NULL CHECK (scope IN ('finsight', 'modelforge')),
    job_id          uuid REFERENCES research_jobs (id),
    stage           text NOT NULL,
    attempt         integer NOT NULL,
    request_sha256  text NOT NULL,
    provider        text NOT NULL,
    model           text NOT NULL,
    status          text NOT NULL CHECK (status IN ('reserved', 'sent', 'completed', 'unknown', 'failed')),
    paid            boolean NOT NULL DEFAULT false,
    reserved_usd    numeric(12, 6) NOT NULL DEFAULT 0,
    cost_usd        numeric(12, 6),
    response        jsonb,
    usage           jsonb,
    latency_ms      integer,
    model_version   text,
    stop_reason     text,
    error           text,
    created_at      timestamptz NOT NULL DEFAULT now(),
    updated_at      timestamptz NOT NULL DEFAULT now(),
    UNIQUE (job_id, stage, attempt)
);
CREATE INDEX model_attempts_paid_open ON model_attempts (status) WHERE paid;

CREATE TABLE spend_limit (
    id          integer PRIMARY KEY CHECK (id = 1),
    limit_usd   numeric(12, 6) NOT NULL
);

CREATE TABLE claims (
    id          bigserial PRIMARY KEY,
    job_id      uuid NOT NULL REFERENCES research_jobs (id),
    finding_key text NOT NULL,
    kind        text NOT NULL,
    status      text NOT NULL,
    payload     jsonb NOT NULL
);

CREATE TABLE reports (
    job_id          uuid PRIMARY KEY REFERENCES research_jobs (id),
    json_path       text NOT NULL,
    markdown_path   text NOT NULL,
    sha256          text NOT NULL,
    created_at      timestamptz NOT NULL DEFAULT now()
);

-- Shared request-rate limit per external host (e.g. SEC: at most 5 requests per second across processes).
CREATE TABLE rate_limits (
    host            text PRIMARY KEY,
    min_interval_ms integer NOT NULL,
    next_allowed_at timestamptz NOT NULL DEFAULT now()
);
