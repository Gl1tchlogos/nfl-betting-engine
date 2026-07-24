import argparse
import io
import json
import logging
import math
import os
from datetime import date, datetime
from typing import Any, Iterable

import httpx
import pandas as pd
from sqlalchemy import MetaData, Table, create_engine, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASE_URL = "https://github.com/nflverse/nflverse-data/releases/download"
SCHEDULES_URL = f"{BASE_URL}/schedules/games.parquet"
PLAYER_STATS_URL = f"{BASE_URL}/stats_player/stats_player_week_{{season}}.parquet"
TEAM_STATS_URL = f"{BASE_URL}/stats_team/stats_team_week_{{season}}.parquet"


def seasons(start: int, end: int) -> Iterable[int]:
    if start > end:
        raise ValueError("start season must be <= end season")
    return range(start, end + 1)


def database_url() -> str:
    value = os.environ.get("DATABASE_URL")
    if not value:
        raise RuntimeError("DATABASE_URL is not set")
    if value.startswith("postgresql://"):
        value = value.replace("postgresql://", "postgresql+psycopg://", 1)
    return value


def read_parquet_url(url: str) -> pd.DataFrame:
    log.info("Downloading %s", url)
    with httpx.Client(timeout=300, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()
    return pd.read_parquet(io.BytesIO(response.content))


def clean_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if isinstance(value, (datetime, date, str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def json_row(row: pd.Series, excluded: set[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in row.items():
        if key in excluded:
            continue
        cleaned = clean_value(value)
        if cleaned is not None:
            result[key] = cleaned
    return result


def records(df: pd.DataFrame) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        output.append({key: clean_value(value) for key, value in row.items()})
    return output


def upsert_rows(engine, table_name: str, rows: list[dict[str, Any]], conflict: list[str]) -> int:
    if not rows:
        return 0
    metadata = MetaData()
    table = Table(table_name, metadata, autoload_with=engine)
    valid_columns = set(table.c.keys())
    rows = [{key: value for key, value in row.items() if key in valid_columns} for row in rows]
    update_columns = [column for column in rows[0] if column not in conflict]
    total = 0
    with engine.begin() as connection:
        for start in range(0, len(rows), 500):
            chunk = rows[start : start + 500]
            statement = pg_insert(table).values(chunk)
            statement = statement.on_conflict_do_update(
                index_elements=conflict,
                set_={column: getattr(statement.excluded, column) for column in update_columns},
            )
            connection.execute(statement)
            total += len(chunk)
    return total


def start_run(engine, dataset: str, season: int | None) -> int:
    with engine.begin() as connection:
        return connection.execute(
            text(
                """
                insert into import_runs (dataset, season)
                values (:dataset, :season)
                returning id
                """
            ),
            {"dataset": dataset, "season": season},
        ).scalar_one()


def finish_run(engine, run_id: int, status: str, row_count: int, error: str | None = None) -> None:
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                update import_runs
                set completed_at = now(), status = :status,
                    rows_processed = :row_count, error_message = :error
                where id = :run_id
                """
            ),
            {"run_id": run_id, "status": status, "row_count": row_count, "error": error},
        )


def prepare_games(raw: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    df = raw[raw["season"].between(start, end)].copy()
    rename = {
        "gameday": "game_date",
        "temp": "temperature",
    }
    df = df.rename(columns=rename)
    if "game_date" in df:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.date
    if "kickoff_time" not in df:
        if "game_date" in df and "gametime" in df:
            combined = df["game_date"].astype(str) + " " + df["gametime"].fillna("00:00")
            df["kickoff_time"] = pd.to_datetime(combined, errors="coerce", utc=True)
        else:
            df["kickoff_time"] = None
    df["source_updated_at"] = datetime.utcnow()
    df["synced_at"] = datetime.utcnow()
    columns = [
        "game_id", "season", "week", "game_type", "game_date", "kickoff_time",
        "weekday", "gametime", "home_team", "away_team", "home_score", "away_score",
        "stadium", "roof", "surface", "temperature", "wind", "location", "overtime",
        "away_rest", "home_rest", "spread_line", "total_line", "home_moneyline",
        "away_moneyline", "away_spread_odds", "home_spread_odds", "under_odds",
        "over_odds", "result", "total", "source_updated_at", "synced_at",
    ]
    present = [column for column in columns if column in df.columns]
    return df[present].dropna(subset=["game_id", "season", "week", "home_team", "away_team"])


def game_lookup(engine, start: int, end: int) -> dict[tuple[int, int, str, str], str]:
    games = Table("games", MetaData(), autoload_with=engine)
    query = select(
        games.c.game_id,
        games.c.season,
        games.c.week,
        games.c.home_team,
        games.c.away_team,
    ).where(games.c.season.between(start, end))
    lookup: dict[tuple[int, int, str, str], str] = {}
    with engine.connect() as connection:
        for row in connection.execute(query).mappings():
            lookup[(row["season"], row["week"], row["home_team"], row["away_team"])] = row["game_id"]
            lookup[(row["season"], row["week"], row["away_team"], row["home_team"])] = row["game_id"]
    return lookup


def attach_game_ids(df: pd.DataFrame, lookup: dict[tuple[int, int, str, str], str]) -> pd.DataFrame:
    if "game_id" in df.columns and df["game_id"].notna().any():
        return df
    df = df.copy()
    df["game_id"] = [
        lookup.get((int(season), int(week), str(team), str(opponent)))
        for season, week, team, opponent in zip(
            df["season"], df["week"], df["team"], df["opponent"]
        )
    ]
    return df


def prepare_players(df: pd.DataFrame) -> pd.DataFrame:
    name_column = "player_display_name" if "player_display_name" in df else "player_name"
    players = pd.DataFrame(
        {
            "player_id": df.get("player_id"),
            "full_name": df.get(name_column),
            "position": df.get("position"),
            "position_group": df.get("position_group"),
            "current_team": df.get("recent_team"),
            "headshot_url": df.get("headshot_url"),
            "updated_at": datetime.utcnow(),
        }
    )
    players["season"] = df.get("season")
    players["week"] = df.get("week")
    players = players.dropna(subset=["player_id", "full_name"])
    players = players.sort_values(["season", "week"]).drop_duplicates("player_id", keep="last")
    return players.drop(columns=["season", "week"])


def prepare_player_stats(raw: pd.DataFrame, lookup: dict[tuple[int, int, str, str], str]) -> pd.DataFrame:
    df = raw.rename(columns={"recent_team": "team", "opponent_team": "opponent"}).copy()
    df = attach_game_ids(df, lookup)
    df = df.dropna(subset=["game_id", "player_id", "season", "week"])
    mapping = {
        "completions": "passing_completions",
        "attempts": "passing_attempts",
        "sacks": "sacks_suffered",
        "sack_yards": "sack_yards_lost",
    }
    df = df.rename(columns=mapping)
    explicit = [
        "game_id", "player_id", "season", "week", "season_type", "team", "opponent",
        "headshot_url", "passing_attempts", "passing_completions", "passing_yards",
        "passing_tds", "interceptions", "passing_air_yards", "passing_yards_after_catch",
        "passing_first_downs", "passing_epa", "passing_cpoe", "sacks_suffered",
        "sack_yards_lost", "carries", "rushing_yards", "rushing_tds",
        "rushing_first_downs", "rushing_epa", "rushing_fumbles",
        "rushing_fumbles_lost", "targets", "receptions", "receiving_yards",
        "receiving_tds", "receiving_air_yards", "receiving_yards_after_catch",
        "receiving_first_downs", "receiving_epa", "receiving_fumbles",
        "receiving_fumbles_lost", "target_share", "air_yards_share", "wopr", "racr",
        "fantasy_points", "fantasy_points_ppr",
    ]
    present = [column for column in explicit if column in df.columns]
    result = df[present].copy()
    excluded = set(present) | {"player_name", "player_display_name", "position", "position_group"}
    result["stats"] = [json.dumps(json_row(row, excluded), default=str) for _, row in df.iterrows()]
    result["synced_at"] = datetime.utcnow()
    return result.drop_duplicates(["game_id", "player_id"], keep="last")


def prepare_team_stats(raw: pd.DataFrame, lookup: dict[tuple[int, int, str, str], str]) -> pd.DataFrame:
    df = raw.rename(columns={"recent_team": "team", "opponent_team": "opponent"}).copy()
    if "team" not in df.columns and "team_abbr" in df.columns:
        df = df.rename(columns={"team_abbr": "team"})
    df = attach_game_ids(df, lookup)
    df = df.dropna(subset=["game_id", "team", "season", "week"])
    mapping = {
        "passing_attempts": "attempts",
        "passing_completions": "completions",
        "interceptions": "passing_interceptions",
    }
    for source, target in mapping.items():
        if source in df.columns and target not in df.columns:
            df = df.rename(columns={source: target})
    explicit = [
        "game_id", "team", "opponent", "season", "week", "season_type",
        "completions", "attempts", "passing_yards", "passing_tds",
        "passing_interceptions", "sacks_suffered", "carries", "rushing_yards",
        "rushing_tds", "targets", "receptions", "receiving_yards", "receiving_tds",
        "passing_epa", "rushing_epa", "receiving_epa", "fantasy_points",
        "fantasy_points_ppr",
    ]
    present = [column for column in explicit if column in df.columns]
    result = df[present].copy()
    excluded = set(present)
    result["stats"] = [json.dumps(json_row(row, excluded), default=str) for _, row in df.iterrows()]
    result["synced_at"] = datetime.utcnow()
    return result.drop_duplicates(["game_id", "team"], keep="last")


def import_schedules(engine, start: int, end: int) -> int:
    run_id = start_run(engine, "schedules", None)
    try:
        rows = records(prepare_games(read_parquet_url(SCHEDULES_URL), start, end))
        count = upsert_rows(engine, "games", rows, ["game_id"])
        finish_run(engine, run_id, "success", count)
        return count
    except Exception as exc:
        finish_run(engine, run_id, "failed", 0, str(exc)[:2000])
        raise


def import_season(engine, season: int, lookup: dict[tuple[int, int, str, str], str]) -> int:
    run_id = start_run(engine, "weekly_stats", season)
    total = 0
    try:
        player_raw = read_parquet_url(PLAYER_STATS_URL.format(season=season))
        players = prepare_players(player_raw)
        total += upsert_rows(engine, "players", records(players), ["player_id"])
        player_stats = prepare_player_stats(player_raw, lookup)
        total += upsert_rows(engine, "player_game_stats", records(player_stats), ["game_id", "player_id"])

        team_raw = read_parquet_url(TEAM_STATS_URL.format(season=season))
        team_stats = prepare_team_stats(team_raw, lookup)
        total += upsert_rows(engine, "team_game_stats", records(team_stats), ["game_id", "team"])

        finish_run(engine, run_id, "success", total)
        return total
    except Exception as exc:
        finish_run(engine, run_id, "failed", total, str(exc)[:2000])
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Import NFLverse schedules and weekly statistics")
    parser.add_argument("--start-season", type=int, default=2016)
    parser.add_argument("--end-season", type=int, default=2025)
    args = parser.parse_args()

    engine = create_engine(database_url(), pool_pre_ping=True)
    schedule_count = import_schedules(engine, args.start_season, args.end_season)
    log.info("Upserted %s games", schedule_count)
    lookup = game_lookup(engine, args.start_season, args.end_season)
    for season in seasons(args.start_season, args.end_season):
        count = import_season(engine, season, lookup)
        log.info("Season %s: upserted %s player/team records", season, count)


if __name__ == "__main__":
    main()
