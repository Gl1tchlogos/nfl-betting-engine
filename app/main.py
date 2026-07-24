from fastapi import Depends, FastAPI, Header, HTTPException, Query
from sqlalchemy import create_engine, text
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str
    api_read_token: str
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
engine = create_engine(settings.database_url, pool_pre_ping=True)
app = FastAPI(title="NFL Betting Engine", version="0.1.0")


def authorize(authorization: str | None = Header(default=None)) -> None:
    expected = f"Bearer {settings.api_read_token}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.get("/health")
def health() -> dict[str, str]:
    with engine.connect() as conn:
        conn.execute(text("select 1"))
    return {"status": "ok"}


@app.get("/players/search", dependencies=[Depends(authorize)])
def search_players(q: str = Query(min_length=2, max_length=80), limit: int = 20):
    sql = text("""
        select player_id, full_name, position, current_team
        from players
        where full_name ilike :q
        order by full_name
        limit :limit
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql, {"q": f"%{q}%", "limit": min(limit, 100)}).mappings().all()
    return list(rows)


@app.get("/players/{player_id}/recent", dependencies=[Depends(authorize)])
def player_recent(player_id: str, games: int = 8):
    sql = text("""
        select season, week, game_id, team, opponent,
               passing_yards, passing_tds, interceptions,
               rushing_yards, rushing_tds,
               receptions, receiving_yards, receiving_tds,
               targets, carries, fantasy_points_ppr
        from player_game_stats
        where player_id = :player_id
        order by season desc, week desc
        limit :games
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql, {"player_id": player_id, "games": min(games, 50)}).mappings().all()
    return list(rows)


@app.get("/games/{game_id}", dependencies=[Depends(authorize)])
def game(game_id: str):
    with engine.connect() as conn:
        row = conn.execute(text("select * from games where game_id=:game_id"), {"game_id": game_id}).mappings().first()
    if not row:
        raise HTTPException(status_code=404, detail="Game not found")
    return dict(row)
