import os

import pytest

TEST_DB = os.environ.get("FINSIGHT_TEST_DATABASE_URL")


@pytest.fixture
def pg():
    """A migrated, empty test database; skipped unless FINSIGHT_TEST_DATABASE_URL is set."""
    if not TEST_DB:
        if os.environ.get("FINSIGHT_REQUIRE_POSTGRES"):     # CI: database tests (e.g. the replay test) must run, never skip
            pytest.fail("FINSIGHT_REQUIRE_POSTGRES is set but FINSIGHT_TEST_DATABASE_URL is not")
        pytest.skip("set FINSIGHT_TEST_DATABASE_URL (e.g. docker compose up -d db)")
    from finsight.storage.db import connect, migrate
    conn = connect(TEST_DB, application_name="finsight-test")
    conn.execute("SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = current_database() "
                 "AND pid <> pg_backend_pid() AND application_name LIKE 'finsight%%'")
    conn.execute("DROP SCHEMA public CASCADE")
    conn.execute("CREATE SCHEMA public")
    migrate(conn)
    yield conn
    conn.close()
