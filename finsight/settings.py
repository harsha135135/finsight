"""Application settings. Every variable uses the FINSIGHT_ prefix except the Anthropic SDK's own."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FINSIGHT_", env_file=".env", extra="ignore")

    state_dir: Path = Path(".finsight")
    database_url: str = "postgresql://finsight:finsight@127.0.0.1:54329/finsight"
    api_key: SecretStr | None = None
    # Read from the environment or .env (pydantic-settings does not export .env to the process environment,
    # where the SDK would look). Never logged.
    anthropic_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("ANTHROPIC_API_KEY", "FINSIGHT_ANTHROPIC_API_KEY"))
    sec_user_agent: str | None = Field(default=None, description="SEC requires 'Name contact@example.com'")

    # Claude spend guard: hard project-wide ceiling until the final evaluation is approved.
    spend_limit_usd: Decimal = Decimal("2.00")

    model_plan: str = "claude-haiku-5-5"
    model_extract: str = "claude-sonnet-5-5"
    model_analyze: str = "claude-sonnet-5-5"
    model_review: str = "claude-sonnet-5-5"
    model_judge: str = "claude-opus-5-5"
    effort_plan: str = "medium"
    effort_default: str = "medium"
    max_tokens_default: int = 16_000

    qwen_base_url: str = "http://127.0.0.1:8000"

    fred_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("FINSIGHT_FRED_API_KEY", "FRED_API_KEY"))   # free key; never logged

    # Dashboard and watcher
    allow_claude: bool = False          # Claude stays off in the dashboard unless explicitly enabled (paid calls)
    watcher_model: str | None = None    # local model for auto-analysis of new filings; None = no auto-analysis
    replay_dir: Path | None = None      # offline replay bundle (SEC responses + recorded model responses); no network


def get_settings() -> Settings:
    return Settings()


def anthropic_client(**kwargs):
    """The official SDK client with SDK retries off (FinSight's ledger decides what is retried)."""
    import anthropic

    key = get_settings().anthropic_api_key
    return anthropic.Anthropic(api_key=key.get_secret_value() if key else None, max_retries=0, **kwargs)
