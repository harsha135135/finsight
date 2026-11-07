"""PostgreSQL connections and schema migrations (psycopg 3).

Connections use the settings LangGraph's PostgresSaver requires (autocommit, no prepared
statements, dict rows), so a worker's single dedicated session can hold its advisory lock,
write application records and back the graph checkpointer.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

MIGRATIONS = Path(__file__).resolve().parents[2] / "migrations"
_MIGRATION_LOCK = 727_001


def connect(url: str, *, application_name: str = "finsight") -> psycopg.Connection:
    return psycopg.connect(url, autocommit=True, prepare_threshold=0, row_factory=dict_row,
                           application_name=application_name)


def migrate(conn: psycopg.Connection, directory: Path = MIGRATIONS) -> list[str]:
    """Apply new migrations in order; refuse if an applied migration file has changed."""
    applied_now = []
    with conn.transaction():
        conn.execute("SELECT pg_advisory_xact_lock(%s)", (_MIGRATION_LOCK,))
        conn.execute("CREATE TABLE IF NOT EXISTS schema_migrations (version text PRIMARY KEY, sha256 text NOT NULL, "
                     "applied_at timestamptz NOT NULL DEFAULT now())")
        applied = {r["version"]: r["sha256"] for r in conn.execute("SELECT version, sha256 FROM schema_migrations")}
        for path in sorted(directory.glob("*.sql")):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if path.stem in applied:
                if applied[path.stem] != digest:
                    raise RuntimeError(f"migration {path.name} changed after it was applied")
                continue
            with psycopg.ClientCursor(conn) as cursor:  # simple query protocol: a file may hold many statements
                cursor.execute(path.read_text())
            conn.execute("INSERT INTO schema_migrations (version, sha256) VALUES (%s, %s)", (path.stem, digest))
            applied_now.append(path.stem)
    return applied_now


def main() -> int:
    """`python -m finsight.storage.db`: apply pending migrations (the Compose `migrate` service)."""
    from finsight.settings import get_settings

    conn = connect(get_settings().database_url, application_name="finsight-migrate")
    print("applied:", migrate(conn) or "none (up to date)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
