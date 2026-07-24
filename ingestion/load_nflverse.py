import argparse
import io
import logging
import math
import os
import time
from datetime import date, datetime, timezone
from typing import Any, Iterable

import httpx
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASE_URL = "https://github.com/nflverse/nflverse-data/releases/download"
SCHEDULES_URL = f"{BASE_URL}/schedules/games.parquet"
PLAYER_STATS_URL = f"{BASE_URL}/stats_player/stats_player_week_{{season}}.parquet"
TEAM_STATS_URL = f"{BASE_URL}/stats_team/stats_team_week_{{season}}.parquet"
DEFAULT_SUPABASE_URL = "https://rhesgemopvbtwgcytapq.supabase.co"


def seasons(start: int, end: int) -> Iterable[int]:
    if start > end:
        raise ValueError("start season must be <= end season")
    return range(start, end + 1)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        output: dict[str, Any] = {}
        for key, item in value.items():
            cleaned = clean_value(item)
            if cleaned is not None:
                output[str(key)] = cleaned
        return output
    if isinstance(value, (list, tuple, set)):
        return [clean_value(item) for item in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    try:
        missing = pd.isna(value)
        if isinstance(missing, bool) and missing:
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime().isoformat()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        try:
            return clean_value(value.item())
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
    return [
        {key: clean_value(value) for key, value in row.items()}
        for row in df.to_dict(orient="records")
    ]


class SupabaseDataAPI:
    def __init__(self) -> None:
        self.base_url = os.environ.get("SUPABASE_URL", DEFAULT_SUPABASE_URL).rstrip("/")
        self.secret_key = os.environ.get("SUPABASE_SECRET_KEY", "").strip()
        if not self.secret_key:
            raise RuntimeError("SUPABASE_SECRET_KEY is not set")
        self.client = httpx.Client(
            base_url=f"{self.base_url}/rest/v1",
            timeout=httpx.Timeout(120.0, connect=20.0),
            follow_redirects=True,
            headers={
                "apikey": self.secret_key,
                "Content-Type": "application/json",
            },
        )

    def close(self) -> None:
        self.client.close()

    def request(
        self,
        method: str,
        table: str,
        *,
        params: dict[str, str] | None = None,
        payload: Any = None,
        prefer: str | None = None,
    ) -> httpx.Response:
        headers = {"Prefer": prefer} if prefer else None
        last_response: httpx.Response | None = None
        for attempt in range(4):
            response = self.client.request(
                method,
                f"/{table}",
                params=params,
                json=payload,
                headers=headers,
            )
            last_response = response
            if response.status_code not in {429, 500, 502, 503, 504}:
                break
            time.sleep(2**attempt)
        assert last_response is not None
        if last_response.is_error:
            detail = last_response.text.replace(self.secret_key, "***")[-2000:]
            raise RuntimeError(
                f"Supabase Data API {method} {table} failed "
                f"({last_response.status_code}): {detail}"
            )
        return last_response

    def healthcheck(self) -> None:
        self.request("GET", "games", params={"select": "game_id", "limit": "1"})

    def upsert(
        self,
        table: str,
        rows: list[dict[str, Any]],
        conflict: list[str],
        chunk_size: int = 200,
    ) -> int:
        if not rows:
            return 0
        total = 0
        for start in range(0, len(rows), chunk_size):
            chunk = rows[start : start + chunk_size]
            self.request(
                "POST",
                table,
                params={"on_conflict": ",".join(conflict)},
                payload=chunk,
                prefer="resolution=merge-duplicates,return=minimal",
            )
            total += len(chunk)
        return total


def read_parquet_url(url: str) -> pd.DataFrame:
    log.info("Downloading %s", url)
    with httpx.Client(timeout=300, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()
    return pd.read_parquet(io.BytesIO(response.content))


def start_run(api: SupabaseDataAPI, dataset: str, season: int | None) -> int:
    response = api.request(
        "POST",
        "import_runs",
        payload={"dataset": dataset, "season": season},
        prefer="return=representation",
    )
    body = response.json()
    if not body or "id" not in body[0]:
        raise RuntimeError("Supabase did not return an import run id")
    return int(body[0]["id"])


def finish_run(
    api: SupabaseDataAPI,
    run_id: int,
    status: str,
    row_count: int,
    error: str | None = None,
) -> None:
    api.request(
        "PATCH",
        "import_runs",
        params={"id": f"eq.{run_id}"},
        payload={
            "completed_at": utc_now(),
            "status": status,
            "rows_processed": row_count,
            "error_message": error,
        },
        prefer="return=minimal",
    )


def prepare_games(raw: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    df = raw[raw["season"].between(start, end)].copy()
    df = df.rename(columns={"gameday": "game_date", "temp": "temperature"})
    if "game_date" in df:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.date
    if "kickoff_time" not in df:
        if "game_date" in df and "gametime" in df:
            combined = df["game_date"].astype(str) + " " + df["gametime"].fillna("00:00")
            df["kickoff_time"] = pd.to_datetime(combined, errors="coerce", utc=True)
        else:
            df["kickoff_time"] = None
    now = utc_now()
    df["source_updated_at"] = now
    df["synced_at"] = now
    columns = [
        "game_id", "season", "week", "game_type", "game_date", "kickoff_time",
        "weekday", "gametime", "home_team", "away_team", "home_score", "away_score",
        "stadium", "roof", "surface", "temperature", "wind", "location", "overtime",
        "away_rest", "home_rest", "spread_line", "total_line", "home_moneyline",
        "away_moneyline", "away_spread_odds", "home_spread_odds", "under_odds",
        "over_odds", "result", "total", "source_updated_at", "synced_at",
    ]
    present = [column for column in columns if column in df.columns]
    return (
        df[present]
        .dropna(subset=["game_id", "season", "week", "home_team", "away_team"])
        .drop_duplicates("game_id", keep="last")
    )


def build_game_lookup(games: pd.DataFrame) -> dict[tuple[int, int, str, str], str]:
    lookup: dict[tuple[int, int, str, str], str] = {}
    for row in games.itertuples(index=False):
        season = int(row.season)
        week = int(row.week)
        home = str(row.home_team)
        away = str(row.away_team)
        game_id = str(row.game_id)
        lookup[(season, week, home, away)] = game_id
        lookup[(season, week, away, home)] = game_id
    return lookup


def attach_game_ids(
    df: pd.DataFrame,
    lookup: dict[tuple[int, int, str, str], str],
) -> pd.DataFrame:
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
            "updated_at": utc_now(),
        }
    )
    players["season"] = df.get("season")
    players["week"] = df.get("week")
    players = players.dropna(subset=["player_id", "full_name"])
    players = players.sort_values(["season", "week"]).drop_duplicates("player_id", keep="last")
    return players.drop(columns=["season", "week"])


def prepare_player_stats(
    raw: pd.DataFrame,
    lookup: dict[tuple[int, int, str, str], str],
) -> pd.DataFrame:
    df = raw.rename(columns={"recent_team": "team", "opponent_team": "opponent"}).copy()
    df = attach_game_ids(df, lookup)
    df = df.dropna(subset=["game_id", "player_id", "season", "week"])
    mapping = {
        "completions": "passing_completions",
        "attempts": "passing_attempts",
        "passing_interceptions": "interceptions",
        "sacks": "sacks_suffered",
        "sack_yards": "sack_yards_lost",
    }
    for source, target in mapping.items():
        if source in df.columns and target not in df.columns:
            df = df.rename(columns={source: target})
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
    result["stats"] = [json_row(row, excluded) for _, row in df.iterrows()]
    result["synced_at"] = utc_now()
    return result.drop_duplicates(["game_id", "player_id"], keep="last")


def prepare_team_stats(
    raw: pd.DataFrame,
    lookup: dict[tuple[int, int, str, str], str],
) -> pd.DataFrame:
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
    result["stats"] = [json_row(row, set(present)) for _, row in df.iterrows()]
    result["synced_at"] = utc_now()
    return result.drop_duplicates(["game_id", "team"], keep="last")


def import_schedules(
    api: SupabaseDataAPI,
    start: int,
    end: int,
) -> tuple[int, dict[tuple[int, int, str, str], str]]:
    run_id = start_run(api, "schedules", None)
    try:
        games = prepare_games(read_parquet_url(SCHEDULES_URL), start, end)
        count = api.upsert("games", records(games), ["game_id"])
        finish_run(api, run_id, "success", count)
        return count, build_game_lookup(games)
    except Exception as exc:
        finish_run(api, run_id, "failed", 0, str(exc)[:2000])
        raise


def import_season(
    api: SupabaseDataAPI,
    season: int,
    lookup: dict[tuple[int, int, str, str], str],
) -> int:
    run_id = start_run(api, "weekly_stats", season)
    total = 0
    try:
        player_raw = read_parquet_url(PLAYER_STATS_URL.format(season=season))
        total += api.upsert("players", records(prepare_players(player_raw)), ["player_id"])
        total += api.upsert(
            "player_game_stats",
            records(prepare_player_stats(player_raw, lookup)),
            ["game_id", "player_id"],
        )

        team_raw = read_parquet_url(TEAM_STATS_URL.format(season=season))
        total += api.upsert(
            "team_game_stats",
            records(prepare_team_stats(team_raw, lookup)),
            ["game_id", "team"],
        )
        finish_run(api, run_id, "success", total)
        return total
    except Exception as exc:
        finish_run(api, run_id, "failed", total, str(exc)[:2000])
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Import NFLverse schedules and weekly statistics")
    parser.add_argument("--start-season", type=int, default=2016)
    parser.add_argument("--end-season", type=int, default=2025)
    args = parser.parse_args()

    api = SupabaseDataAPI()
    try:
        api.healthcheck()
        schedule_count, lookup = import_schedules(api, args.start_season, args.end_season)
        log.info("Upserted %s games", schedule_count)
        for season in seasons(args.start_season, args.end_season):
            count = import_season(api, season, lookup)
            log.info("Season %s: upserted %s player/team records", season, count)
    finally:
        api.close()


if __name__ == "__main__":
    main()
