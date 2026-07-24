# NFL Betting Engine

Starter backend for a live NFL analytics database used for PrizePicks and Underdog analysis.

## Initial scope
- NFL seasons 2016-2025, plus 2026 updates
- Games, weekly player stats, rosters and play-by-play imports
- PostgreSQL/Supabase storage
- Read-only FastAPI endpoints
- Rolling player and team summaries

## Setup
1. Copy `.env.example` to `.env` locally.
2. Add your Supabase/Postgres connection string to `.env`.
3. Run `sql/001_initial_schema.sql` in the Supabase SQL editor.
4. Install Python 3.11+ and run `pip install -e .`.
5. Run `python -m ingestion.load_nflverse --start-season 2016 --end-season 2025`.
6. Start the API with `uvicorn app.main:app --reload`.

Never commit `.env`, database passwords, service-role keys, or API keys.
