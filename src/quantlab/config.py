"""Typed configuration loaded from environment variables (and an optional ``.env`` file).

Secrets are only ever read from the environment. Nothing in this module has defaults that
contain credentials.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    data_dir: Path = Field(default=Path("./data"), alias="QUANTLAB_DATA_DIR")

    fred_api_key: SecretStr | None = Field(default=None, alias="FRED_API_KEY")
    tiingo_api_key: SecretStr | None = Field(default=None, alias="TIINGO_API_KEY")
    edgar_user_agent: str | None = Field(default=None, alias="EDGAR_USER_AGENT")
    orats_api_key: SecretStr | None = Field(default=None, alias="ORATS_API_KEY")
    polygon_api_key: SecretStr | None = Field(default=None, alias="POLYGON_API_KEY")
    norgate_export_dir: Path | None = Field(default=None, alias="NORGATE_EXPORT_DIR")

    bar_publication_delay_min: int = Field(default=15, alias="QUANTLAB_BAR_PUBLICATION_DELAY_MIN")
    default_seed: int = Field(default=20260927, alias="QUANTLAB_DEFAULT_SEED")

    live_trading_enabled: bool = Field(default=False, alias="QUANTLAB_LIVE_TRADING_ENABLED")
    live_safety_review_file: Path | None = Field(default=None, alias="QUANTLAB_LIVE_SAFETY_REVIEW_FILE")

    @field_validator(
        "fred_api_key", "tiingo_api_key", "edgar_user_agent", "orats_api_key",
        "polygon_api_key", "norgate_export_dir", "live_safety_review_file", mode="before",
    )
    @classmethod
    def _blank_is_unset(cls, v: object) -> object:
        # `.env.example` ships keys with empty values; an empty string means "not configured".
        if isinstance(v, str) and not v.strip():
            return None
        return v

    @property
    def store_dir(self) -> Path:
        return self.data_dir / "store"

    @property
    def registry_path(self) -> Path:
        return self.data_dir / "registry.sqlite"

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
