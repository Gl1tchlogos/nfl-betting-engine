create extension if not exists pg_trgm;

create table if not exists teams (
  team_abbr text primary key,
  team_name text not null,
  conference text,
  division text
);

create table if not exists players (
  player_id text primary key,
  full_name text not null,
  position text,
  current_team text,
  birth_date date,
  height_inches integer,
  weight_lbs integer,
  years_experience integer,
  updated_at timestamptz not null default now()
);
create index if not exists players_name_trgm_idx on players using gin (full_name gin_trgm_ops);

create table if not exists games (
  game_id text primary key,
  season integer not null,
  week integer not null,
  game_type text,
  game_date date,
  kickoff_time timestamptz,
  home_team text not null,
  away_team text not null,
  home_score integer,
  away_score integer,
  stadium text,
  roof text,
  surface text,
  temperature numeric,
  wind numeric,
  spread_line numeric,
  total_line numeric,
  home_moneyline integer,
  away_moneyline integer,
  result integer,
  total integer,
  created_at timestamptz not null default now()
);
create index if not exists games_season_week_idx on games (season, week);

create table if not exists player_game_stats (
  game_id text not null references games(game_id) on delete cascade,
  player_id text not null references players(player_id) on delete cascade,
  season integer not null,
  week integer not null,
  team text,
  opponent text,
  passing_attempts numeric,
  passing_completions numeric,
  passing_yards numeric,
  passing_tds numeric,
  interceptions numeric,
  carries numeric,
  rushing_yards numeric,
  rushing_tds numeric,
  targets numeric,
  receptions numeric,
  receiving_yards numeric,
  receiving_tds numeric,
  fantasy_points_ppr numeric,
  primary key (game_id, player_id)
);
create index if not exists pgs_player_recent_idx on player_game_stats (player_id, season desc, week desc);

create table if not exists plays (
  play_id bigint generated always as identity primary key,
  game_id text not null references games(game_id) on delete cascade,
  source_play_id text,
  qtr integer,
  game_seconds_remaining integer,
  posteam text,
  defteam text,
  down integer,
  ydstogo integer,
  yardline_100 integer,
  play_type text,
  yards_gained numeric,
  epa numeric,
  wpa numeric,
  success numeric,
  passer_player_id text,
  rusher_player_id text,
  receiver_player_id text,
  desc_text text,
  unique (game_id, source_play_id)
);
create index if not exists plays_game_idx on plays (game_id);
create index if not exists plays_players_idx on plays (passer_player_id, rusher_player_id, receiver_player_id);

create table if not exists odds_snapshots (
  id bigint generated always as identity primary key,
  provider text not null,
  platform text not null,
  market text not null,
  player_id text,
  game_id text,
  line_value numeric,
  over_price integer,
  under_price integer,
  observed_at timestamptz not null,
  raw_payload jsonb,
  unique (provider, platform, market, player_id, game_id, observed_at)
);
create index if not exists odds_lookup_idx on odds_snapshots (platform, market, player_id, observed_at desc);

create table if not exists model_predictions (
  id bigint generated always as identity primary key,
  model_version text not null,
  game_id text,
  player_id text,
  market text not null,
  line_value numeric,
  projected_mean numeric,
  projected_median numeric,
  over_probability numeric,
  fair_over_odds integer,
  created_at timestamptz not null default now()
);

alter table teams enable row level security;
alter table players enable row level security;
alter table games enable row level security;
alter table player_game_stats enable row level security;
alter table plays enable row level security;
alter table odds_snapshots enable row level security;
alter table model_predictions enable row level security;
