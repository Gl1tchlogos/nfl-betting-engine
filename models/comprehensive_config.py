from __future__ import annotations

MODEL_VERSION = "comprehensive_v1"
DEVELOPMENT_END_SEASON = 2022
VALIDATION_SEASON = 2023
TEST_SEASONS = [2024, 2025]
MIN_TRAIN_ROWS = 300
MIN_POSITION_TRAIN_ROWS = 250
MIN_POSITION_VALIDATION_ROWS = 40

COMMON_ID_COLUMNS = [
    "game_id", "player_id", "full_name", "position", "position_group",
    "season", "week", "season_type", "team", "opponent", "game_date",
    "game_ts", "is_home", "prior_games", "birth_date", "years_experience",
    "height_inches", "weight_lbs", "days_since_previous_game", "roof",
    "surface", "temperature", "wind",
]

COMMON_FEATURE_GROUPS = {
    "environment": [
        "week", "is_home", "team_spread", "total_line", "implied_team_points",
        "implied_opponent_points", "team_rest", "opponent_rest", "rest_difference",
        "is_indoor", "high_wind", "freezing_temperature", "surface_grass",
        "temperature", "wind", "is_postseason", "late_season", "team_favored",
    ],
    "continuity": [
        "prior_games", "days_since_previous_game", "changed_team",
        "return_after_14_days", "age_years", "years_experience",
        "height_inches", "weight_lbs",
    ],
}

PASSING_GROUPS = {
    **COMMON_FEATURE_GROUPS,
    "trend": [
        "passing_attempts_avg_3", "passing_attempts_avg_5",
        "passing_attempts_ewm_5", "passing_attempts_trend_3_vs_5",
        "passing_yards_avg_3", "passing_yards_avg_5", "passing_yards_avg_10",
        "passing_yards_ewm_5", "passing_yards_sd_10",
        "passing_yards_trend_3_vs_10", "passing_yards_shrunk_baseline",
    ],
    "opportunity": [
        "passing_attempt_share_ewm_5", "team_attempts_avg_5",
        "team_passing_air_yards_avg_5", "team_passing_first_downs_avg_5",
    ],
    "efficiency": [
        "passing_air_yards_avg_5", "passing_yac_avg_5",
        "passing_first_downs_avg_5", "passing_epa_avg_5",
        "passing_cpoe_avg_5", "sacks_suffered_avg_5",
        "passing_20_plus_avg_5", "passing_yards_per_attempt_avg_5",
        "team_passing_yac_avg_5", "team_passing_cpoe_avg_5",
        "team_sacks_suffered_avg_5", "team_passing_20_plus_avg_5",
    ],
    "matchup": [
        "opponent_defensive_sacks_avg_5", "opponent_defensive_qb_hits_avg_5",
        "opponent_defensive_interceptions_avg_5",
        "opponent_defensive_passes_defended_avg_5",
        "position_passing_attempts_allowed_avg_5",
        "position_passing_yards_allowed_avg_5",
        "position_passing_yards_per_attempt_allowed_avg_5",
    ],
}

RUSHING_GROUPS = {
    **COMMON_FEATURE_GROUPS,
    "trend": [
        "carries_avg_3", "carries_avg_5", "carries_ewm_5",
        "carries_trend_3_vs_5", "rushing_yards_avg_3",
        "rushing_yards_avg_5", "rushing_yards_avg_10",
        "rushing_yards_ewm_5", "rushing_yards_sd_10",
        "rushing_yards_trend_3_vs_10", "rushing_yards_shrunk_baseline",
    ],
    "opportunity": [
        "carry_share_avg_3", "carry_share_avg_5", "carry_share_ewm_5",
        "team_carries_avg_5", "team_rushing_first_downs_avg_5",
    ],
    "efficiency": [
        "rushing_first_downs_avg_5", "rushing_epa_avg_5",
        "rushing_fumbles_lost_avg_5", "rushing_20_plus_avg_5",
        "rushing_yards_per_carry_avg_5", "team_rushing_20_plus_avg_5",
        "team_penalty_yards_avg_5",
    ],
    "matchup": [
        "opponent_defensive_tackles_for_loss_avg_5",
        "opponent_defensive_fumbles_forced_avg_5",
        "position_carries_allowed_avg_5",
        "position_rushing_yards_allowed_avg_3",
        "position_rushing_yards_allowed_avg_5",
        "position_rushing_yards_allowed_avg_10",
        "position_rushing_yards_per_carry_allowed_avg_5",
        "position_rushing_20_plus_allowed_avg_5",
    ],
}

RECEIVING_GROUPS = {
    **COMMON_FEATURE_GROUPS,
    "trend": [
        "targets_avg_3", "targets_avg_5", "targets_ewm_5",
        "targets_trend_3_vs_5", "receptions_avg_3", "receptions_avg_5",
        "receptions_ewm_5", "receptions_trend_3_vs_5",
        "receiving_yards_avg_3", "receiving_yards_avg_5",
        "receiving_yards_avg_10", "receiving_yards_ewm_5",
        "receiving_yards_sd_10", "receiving_yards_trend_3_vs_10",
        "receiving_yards_shrunk_baseline",
    ],
    "opportunity": [
        "target_share_avg_3", "target_share_avg_5", "target_share_ewm_5",
        "air_yards_share_avg_5", "wopr_avg_5", "reception_share_avg_5",
        "receiving_yards_share_avg_5", "team_attempts_avg_5",
        "team_passing_air_yards_avg_5",
    ],
    "efficiency": [
        "receiving_air_yards_avg_5", "receiving_yac_avg_5",
        "receiving_first_downs_avg_5", "receiving_epa_avg_5",
        "receiving_fumbles_lost_avg_5", "receiving_20_plus_avg_5",
        "racr_avg_5", "receiving_yards_per_target_avg_5",
        "receptions_per_target_avg_5", "team_passing_yac_avg_5",
        "team_passing_20_plus_avg_5",
    ],
    "matchup": [
        "opponent_defensive_sacks_avg_5", "opponent_defensive_qb_hits_avg_5",
        "opponent_defensive_interceptions_avg_5",
        "opponent_defensive_passes_defended_avg_5",
        "position_targets_allowed_avg_3", "position_targets_allowed_avg_5",
        "position_targets_allowed_avg_10", "position_receptions_allowed_avg_5",
        "position_receiving_yards_allowed_avg_3",
        "position_receiving_yards_allowed_avg_5",
        "position_receiving_yards_allowed_avg_10",
        "position_receiving_air_yards_allowed_avg_5",
        "position_receiving_yac_allowed_avg_5",
        "position_receiving_yards_per_target_allowed_avg_5",
        "position_receiving_20_plus_allowed_avg_5",
    ],
}

PLAYER_TARGETS = {
    "passing_attempts": {
        "domain": "passing", "target": "passing_attempts",
        "baseline": "passing_attempts_shrunk_baseline",
        "eligibility": ("passing_attempts_avg_3", 10.0),
        "groups": PASSING_GROUPS,
    },
    "passing_yards": {
        "domain": "passing", "target": "passing_yards",
        "baseline": "passing_yards_shrunk_baseline",
        "eligibility": ("passing_attempts_avg_3", 10.0),
        "groups": PASSING_GROUPS,
    },
    "carries": {
        "domain": "rushing", "target": "carries",
        "baseline": "carries_shrunk_baseline",
        "eligibility": ("carries_avg_3", 4.0),
        "groups": RUSHING_GROUPS,
    },
    "rushing_yards": {
        "domain": "rushing", "target": "rushing_yards",
        "baseline": "rushing_yards_shrunk_baseline",
        "eligibility": ("carries_avg_3", 4.0),
        "groups": RUSHING_GROUPS,
    },
    "targets": {
        "domain": "receiving", "target": "targets",
        "baseline": "targets_shrunk_baseline",
        "eligibility": ("targets_avg_3", 3.0),
        "groups": RECEIVING_GROUPS,
    },
    "receptions": {
        "domain": "receiving", "target": "receptions",
        "baseline": "receptions_shrunk_baseline",
        "eligibility": ("targets_avg_3", 3.0),
        "groups": RECEIVING_GROUPS,
    },
    "receiving_yards": {
        "domain": "receiving", "target": "receiving_yards",
        "baseline": "receiving_yards_shrunk_baseline",
        "eligibility": ("targets_avg_3", 3.0),
        "groups": RECEIVING_GROUPS,
    },
}

GAME_ID_COLUMNS = [
    "game_id", "season", "week", "game_type", "game_date", "kickoff_time",
    "home_team", "away_team", "actual_home_margin", "actual_total",
    "spread_line", "total_line",
]

GAME_FEATURE_GROUPS = {
    "market_environment": [
        "week", "spread_line", "total_line", "implied_home_points",
        "implied_away_points", "home_rest", "away_rest", "rest_difference",
        "absolute_spread", "is_indoor", "high_wind", "freezing_temperature",
        "temperature", "wind", "is_postseason", "late_season",
    ],
    "recent_form": [
        "home_prior_games", "home_points_for_avg_3", "home_points_for_avg_5",
        "home_points_for_avg_10", "home_points_allowed_avg_5",
        "home_passing_yards_avg_5", "home_rushing_yards_avg_5",
        "home_offensive_volume_avg_5", "home_passing_epa_avg_5",
        "home_rushing_epa_avg_5", "home_scoring_trend_3_vs_10",
        "away_prior_games", "away_points_for_avg_3", "away_points_for_avg_5",
        "away_points_for_avg_10", "away_points_allowed_avg_5",
        "away_passing_yards_avg_5", "away_rushing_yards_avg_5",
        "away_offensive_volume_avg_5", "away_passing_epa_avg_5",
        "away_rushing_epa_avg_5", "away_scoring_trend_3_vs_10",
    ],
    "matchup": [
        "home_passing_yards_allowed_avg_5", "home_rushing_yards_allowed_avg_5",
        "away_passing_yards_allowed_avg_5", "away_rushing_yards_allowed_avg_5",
        "home_passing_epa_allowed_avg_5", "home_rushing_epa_allowed_avg_5",
        "away_passing_epa_allowed_avg_5", "away_rushing_epa_allowed_avg_5",
        "home_scoring_matchup_edge", "away_scoring_matchup_edge",
        "home_passing_matchup_edge", "away_passing_matchup_edge",
        "home_rushing_matchup_edge", "away_rushing_matchup_edge",
    ],
    "advanced_offense": [
        "home_attempts_avg_5", "home_carries_avg_5",
        "home_passing_air_yards_avg_5", "home_passing_yac_avg_5",
        "home_passing_first_downs_avg_5", "home_passing_cpoe_avg_5",
        "home_sacks_suffered_avg_5", "home_rushing_first_downs_avg_5",
        "home_passing_20_plus_avg_5", "home_rushing_20_plus_avg_5",
        "away_attempts_avg_5", "away_carries_avg_5",
        "away_passing_air_yards_avg_5", "away_passing_yac_avg_5",
        "away_passing_first_downs_avg_5", "away_passing_cpoe_avg_5",
        "away_sacks_suffered_avg_5", "away_rushing_first_downs_avg_5",
        "away_passing_20_plus_avg_5", "away_rushing_20_plus_avg_5",
    ],
    "discipline_turnovers": [
        "home_fumbles_lost_avg_5", "away_fumbles_lost_avg_5",
        "home_penalty_yards_avg_5", "away_penalty_yards_avg_5",
        "penalty_yards_difference", "fumbles_lost_difference",
    ],
    "defensive_pressure": [
        "home_defensive_sacks_avg_5", "home_defensive_qb_hits_avg_5",
        "home_defensive_interceptions_avg_5",
        "home_defensive_passes_defended_avg_5",
        "home_defensive_tackles_for_loss_avg_5",
        "home_defensive_fumbles_forced_avg_5",
        "away_defensive_sacks_avg_5", "away_defensive_qb_hits_avg_5",
        "away_defensive_interceptions_avg_5",
        "away_defensive_passes_defended_avg_5",
        "away_defensive_tackles_for_loss_avg_5",
        "away_defensive_fumbles_forced_avg_5",
        "defensive_qb_hits_difference",
    ],
}

GAME_TARGETS = {
    "game_margin": {
        "target": "actual_home_margin", "baseline": "spread_line",
        "groups": GAME_FEATURE_GROUPS,
    },
    "game_total": {
        "target": "actual_total", "baseline": "total_line",
        "groups": GAME_FEATURE_GROUPS,
    },
}
