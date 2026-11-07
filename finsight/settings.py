"""Application settings. Every variable uses the FINSIGHT_ prefix except the Anthropic SDK's own."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


    model: str
    effort: str


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FINSIGHT_", env_file=".env", extra="ignore")

    state_dir: Path = Path(".finsight")
    database_url: str = "postgresql://finsight:finsight@127.0.0.1:54329/finsight"
    api_key: SecretStr | None = None
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


def get_settings() -> Settings:
    return Settings()
