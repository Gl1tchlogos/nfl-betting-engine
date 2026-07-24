import argparse
import io
import logging
from typing import Iterable

import httpx
import pandas as pd
from sqlalchemy import create_engine
from pydantic_settings import BaseSettings, SettingsConfigDict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


class Settings(BaseSettings):
    database_url: str
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


def seasons(start: int, end: int) -> Iterable[int]:
    if start > end:
        raise ValueError("start season must be <= end season")
    return range(start, end + 1)


def read_parquet_url(url: str) -> pd.DataFrame:
    log.info("Downloading %s", url)
    with httpx.Client(timeout=180, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()
    return pd.read_parquet(io.BytesIO(response.content))


def load_weekly(engine, season: int) -> None:
    url = f"https://github.com/nflverse/nflverse-data/releases/download/player_stats/player_stats_{season}.parquet"
    df = read_parquet_url(url)
    rename = {
        "player_display_name": "full_name",
        "recent_team": "team",
        "opponent_team": "opponent",
    }
    df = df.rename(columns=rename)
    required = ["player_id", "full_name", "position", "team"]
    players = df[[c for c in required if c in df.columns]].dropna(subset=["player_id"]).drop_duplicates("player_id")
    players = players.rename(columns={"team": "current_team"})
    players.to_sql("players", engine, if_exists="append", index=False, method="multi", chunksize=1000)

    cols = [
        "player_id", "season", "week", "team", "opponent",
        "passing_attempts", "completions", "passing_yards", "passing_tds", "interceptions",
        "carries", "rushing_yards", "rushing_tds", "targets", "receptions",
        "receiving_yards", "receiving_tds", "fantasy_points_ppr", "game_id",
    ]
    stats = df[[c for c in cols if c in df.columns]].copy()
    stats = stats.rename(columns={"completions": "passing_completions"})
    stats.to_sql("player_game_stats", engine, if_exists="append", index=False, method="multi", chunksize=2000)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-season", type=int, default=2016)
    parser.add_argument("--end-season", type=int, default=2025)
    args = parser.parse_args()
    settings = Settings()
    engine = create_engine(settings.database_url, pool_pre_ping=True)
    for season in seasons(args.start_season, args.end_season):
        try:
            load_weekly(engine, season)
        except Exception:
            log.exception("Failed loading season %s", season)
            raise


if __name__ == "__main__":
    main()
