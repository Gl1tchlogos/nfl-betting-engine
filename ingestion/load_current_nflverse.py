from __future__ import annotations

import argparse
import logging

import httpx

from ingestion.load_nflverse import (
    PLAYER_STATS_URL,
    TEAM_STATS_URL,
    SupabaseDataAPI,
    import_schedules,
    import_season,
    seasons,
)

log = logging.getLogger(__name__)


def dataset_is_available(url: str) -> bool:
    """Return False only when NFLverse has not published the requested file yet."""
    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        response = client.head(url)
        if response.status_code == 404:
            return False
        response.raise_for_status()
        return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import current NFLverse data without failing on unpublished weekly files"
    )
    parser.add_argument("--start-season", type=int, required=True)
    parser.add_argument("--end-season", type=int, required=True)
    args = parser.parse_args()

    api = SupabaseDataAPI()
    try:
        api.healthcheck()
        schedule_count, lookup = import_schedules(api, args.start_season, args.end_season)
        log.info("Upserted %s games", schedule_count)

        for season in seasons(args.start_season, args.end_season):
            player_url = PLAYER_STATS_URL.format(season=season)
            team_url = TEAM_STATS_URL.format(season=season)
            missing = [
                label
                for label, url in (("player", player_url), ("team", team_url))
                if not dataset_is_available(url)
            ]
            if missing:
                log.warning(
                    "NFLverse %s weekly statistics for %s are not published yet; "
                    "schedule import succeeded and the stats import was skipped cleanly.",
                    " and ".join(missing),
                    season,
                )
                continue

            count = import_season(api, season, lookup)
            log.info("Season %s: upserted %s player/team records", season, count)
    finally:
        api.close()


if __name__ == "__main__":
    main()
