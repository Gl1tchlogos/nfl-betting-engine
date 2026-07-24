from __future__ import annotations

import argparse
import json
import math
import os
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_VERSION = "baseline_v1"
VALIDATION_SEASON = 2023
TEST_SEASONS = [2024, 2025]
PAGE_SIZE = 1000
BLEND_WEIGHTS = [0.25, 0.50, 0.75, 1.00]
BET_THRESHOLDS = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]


GAME_FEATURES = [
    "week",
    "spread_line",
    "total_line",
    "home_rest",
    "away_rest",
    "home_prior_games",
    "home_points_for_avg_3",
    "home_points_for_avg_5",
    "home_points_for_avg_10",
    "home_points_allowed_avg_5",
    "home_passing_yards_avg_5",
    "home_rushing_yards_avg_5",
    "home_offensive_volume_avg_5",
    "home_passing_epa_avg_5",
    "home_rushing_epa_avg_5",
    "home_passing_yards_allowed_avg_5",
    "home_rushing_yards_allowed_avg_5",
    "home_passing_epa_allowed_avg_5",
    "home_rushing_epa_allowed_avg_5",
    "home_scoring_trend_3_vs_10",
    "away_prior_games",
    "away_points_for_avg_3",
    "away_points_for_avg_5",
    "away_points_for_avg_10",
    "away_points_allowed_avg_5",
    "away_passing_yards_avg_5",
    "away_rushing_yards_avg_5",
    "away_offensive_volume_avg_5",
    "away_passing_epa_avg_5",
    "away_rushing_epa_avg_5",
    "away_passing_yards_allowed_avg_5",
    "away_rushing_yards_allowed_avg_5",
    "away_passing_epa_allowed_avg_5",
    "away_rushing_epa_allowed_avg_5",
    "away_scoring_trend_3_vs_10",
    "home_scoring_matchup_edge",
    "away_scoring_matchup_edge",
    "home_passing_matchup_edge",
    "away_passing_matchup_edge",
    "home_rushing_matchup_edge",
    "away_rushing_matchup_edge",
    "rest_difference",
    "abs_spread_line",
    "is_postseason",
]

GAME_SELECT = [
    "game_id",
    "season",
    "week",
    "game_type",
    "spread_line",
    "total_line",
    "home_rest",
    "away_rest",
    "actual_home_margin",
    "actual_total",
] + [column for column in GAME_FEATURES if column not in {
    "week",
    "spread_line",
    "total_line",
    "home_rest",
    "away_rest",
    "rest_difference",
    "abs_spread_line",
    "is_postseason",
}]

COMMON_PROP_SELECT = [
    "game_id",
    "player_id",
    "full_name",
    "position",
    "position_group",
    "season",
    "week",
    "is_home",
    "prior_games",
    "days_since_previous_game",
    "team_spread",
    "total_line",
    "team_rest",
    "opponent_rest",
    "team_points_for_avg_5",
    "team_passing_yards_avg_5",
    "team_rushing_yards_avg_5",
    "team_offensive_volume_avg_5",
    "team_passing_epa_avg_5",
    "team_rushing_epa_avg_5",
    "opponent_points_allowed_avg_5",
]

PROP_CONFIGS = {
    "passing_yards": {
        "target": "actual_passing_yards",
        "baseline": "passing_yards_avg_5",
        "filters": {
            "prior_games": "gte.3",
            "passing_attempts_avg_3": "gte.10",
            "actual_passing_yards": "not.is.null",
        },
        "features": [
            "week", "is_home", "prior_games", "days_since_previous_game",
            "team_spread", "total_line", "team_rest", "opponent_rest",
            "passing_attempts_avg_3", "passing_attempts_avg_5", "passing_attempts_avg_10",
            "passing_yards_avg_3", "passing_yards_avg_5", "passing_yards_avg_10",
            "passing_yards_sd_10", "passing_tds_avg_5", "interceptions_avg_5",
            "passing_yards_trend_3_vs_10", "team_points_for_avg_5",
            "team_passing_yards_avg_5", "team_offensive_volume_avg_5",
            "team_passing_epa_avg_5", "opponent_points_allowed_avg_5",
            "opponent_passing_yards_allowed_avg_3",
            "opponent_passing_yards_allowed_avg_5",
            "opponent_passing_yards_allowed_avg_10",
            "opponent_pass_attempts_avg_5", "passing_epa_allowed_avg_5",
            "passing_defense_trend_3_vs_10", "rest_difference",
            "pos_qb", "pos_rb", "pos_wr", "pos_te",
        ],
        "select": [
            "passing_attempts_avg_3", "passing_attempts_avg_5", "passing_attempts_avg_10",
            "passing_yards_avg_3", "passing_yards_avg_5", "passing_yards_avg_10",
            "passing_yards_sd_10", "passing_tds_avg_5", "interceptions_avg_5",
            "passing_yards_trend_3_vs_10",
            "opponent_passing_yards_allowed_avg_3",
            "opponent_passing_yards_allowed_avg_5",
            "opponent_passing_yards_allowed_avg_10",
            "opponent_pass_attempts_avg_5", "passing_epa_allowed_avg_5",
            "passing_defense_trend_3_vs_10", "actual_passing_yards",
        ],
    },
    "rushing_yards": {
        "target": "actual_rushing_yards",
        "baseline": "rushing_yards_avg_5",
        "filters": {
            "prior_games": "gte.3",
            "carries_avg_3": "gte.4",
            "actual_rushing_yards": "not.is.null",
        },
        "features": [
            "week", "is_home", "prior_games", "days_since_previous_game",
            "team_spread", "total_line", "team_rest", "opponent_rest",
            "carries_avg_3", "carries_avg_5", "carries_avg_10",
            "rushing_yards_avg_3", "rushing_yards_avg_5", "rushing_yards_avg_10",
            "rushing_yards_sd_10", "rushing_yards_trend_3_vs_10",
            "team_points_for_avg_5", "team_rushing_yards_avg_5",
            "team_offensive_volume_avg_5", "team_rushing_epa_avg_5",
            "opponent_points_allowed_avg_5",
            "opponent_rushing_yards_allowed_avg_3",
            "opponent_rushing_yards_allowed_avg_5",
            "opponent_rushing_yards_allowed_avg_10",
            "opponent_carries_avg_5", "rushing_epa_allowed_avg_5",
            "rushing_defense_trend_3_vs_10", "rest_difference",
            "pos_qb", "pos_rb", "pos_wr", "pos_te",
        ],
        "select": [
            "carries_avg_3", "carries_avg_5", "carries_avg_10",
            "rushing_yards_avg_3", "rushing_yards_avg_5", "rushing_yards_avg_10",
            "rushing_yards_sd_10", "rushing_yards_trend_3_vs_10",
            "opponent_rushing_yards_allowed_avg_3",
            "opponent_rushing_yards_allowed_avg_5",
            "opponent_rushing_yards_allowed_avg_10",
            "opponent_carries_avg_5", "rushing_epa_allowed_avg_5",
            "rushing_defense_trend_3_vs_10", "actual_rushing_yards",
        ],
    },
    "receiving_yards": {
        "target": "actual_receiving_yards",
        "baseline": "receiving_yards_avg_5",
        "filters": {
            "prior_games": "gte.3",
            "targets_avg_3": "gte.3",
            "actual_receiving_yards": "not.is.null",
        },
        "features": [
            "week", "is_home", "prior_games", "days_since_previous_game",
            "team_spread", "total_line", "team_rest", "opponent_rest",
            "targets_avg_3", "targets_avg_5", "targets_avg_10", "receptions_avg_5",
            "receiving_yards_avg_3", "receiving_yards_avg_5", "receiving_yards_avg_10",
            "receiving_yards_sd_10", "target_share_avg_5", "air_yards_share_avg_5",
            "wopr_avg_5", "receiving_yards_trend_3_vs_10", "targets_trend_3_vs_10",
            "team_points_for_avg_5", "team_passing_yards_avg_5",
            "team_offensive_volume_avg_5", "team_passing_epa_avg_5",
            "opponent_points_allowed_avg_5",
            "opponent_passing_yards_allowed_avg_3",
            "opponent_passing_yards_allowed_avg_5",
            "opponent_passing_yards_allowed_avg_10",
            "opponent_pass_attempts_avg_5", "passing_epa_allowed_avg_5",
            "passing_defense_trend_3_vs_10", "rest_difference",
            "pos_qb", "pos_rb", "pos_wr", "pos_te",
        ],
        "select": [
            "targets_avg_3", "targets_avg_5", "targets_avg_10", "receptions_avg_5",
            "receiving_yards_avg_3", "receiving_yards_avg_5", "receiving_yards_avg_10",
            "receiving_yards_sd_10", "target_share_avg_5", "air_yards_share_avg_5",
            "wopr_avg_5", "receiving_yards_trend_3_vs_10", "targets_trend_3_vs_10",
            "opponent_passing_yards_allowed_avg_3",
            "opponent_passing_yards_allowed_avg_5",
            "opponent_passing_yards_allowed_avg_10",
            "opponent_pass_attempts_avg_5", "passing_epa_allowed_avg_5",
            "passing_defense_trend_3_vs_10", "actual_receiving_yards",
        ],
    },
}


@dataclass
class Selection:
    algorithm: str
    blend_weight: float
    validation_mae: float
    validation_baseline_mae: float
    validation_predictions: np.ndarray


class SupabaseClient:
    def __init__(self) -> None:
        base_url = os.environ.get("SUPABASE_URL", "").rstrip("/")
        secret_key = os.environ.get("SUPABASE_SECRET_KEY", "").strip()
        if not base_url or not secret_key:
            raise RuntimeError("SUPABASE_URL and SUPABASE_SECRET_KEY are required")
        self.secret_key = secret_key
        self.client = httpx.Client(
            base_url=f"{base_url}/rest/v1",
            headers={"apikey": secret_key, "Content-Type": "application/json"},
            timeout=httpx.Timeout(180.0, connect=30.0),
            follow_redirects=True,
        )

    def close(self) -> None:
        self.client.close()

    def _check(self, response: httpx.Response) -> httpx.Response:
        if response.is_error:
            detail = response.text.replace(self.secret_key, "***")[-2000:]
            raise RuntimeError(f"Supabase request failed ({response.status_code}): {detail}")
        return response

    def fetch_all(
        self,
        table: str,
        columns: list[str],
        filters: dict[str, str] | None = None,
        order: str = "season.asc,week.asc",
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        offset = 0
        filters = filters or {}
        while True:
            params: dict[str, str | int] = {
                "select": ",".join(dict.fromkeys(columns)),
                "limit": PAGE_SIZE,
                "offset": offset,
                "order": order,
            }
            params.update(filters)
            response = self._check(self.client.get(f"/{table}", params=params))
            page = response.json()
            if not page:
                break
            rows.extend(page)
            if len(page) < PAGE_SIZE:
                break
            offset += PAGE_SIZE
        return pd.DataFrame(rows)

    def start_run(self) -> int:
        response = self._check(
            self.client.post(
                "/model_backtest_runs",
                json={
                    "model_version": MODEL_VERSION,
                    "status": "running",
                    "train_through_season": VALIDATION_SEASON - 1,
                    "validation_season": VALIDATION_SEASON,
                    "test_seasons": TEST_SEASONS,
                },
                headers={"Prefer": "return=representation"},
            )
        )
        body = response.json()
        return int(body[0]["id"])

    def finish_run(
        self,
        run_id: int,
        status: str,
        report: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        payload = {
            "status": status,
            "report": report,
            "error_message": error,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        self._check(
            self.client.patch(
                "/model_backtest_runs",
                params={"id": f"eq.{run_id}"},
                json=payload,
                headers={"Prefer": "return=minimal"},
            )
        )


def make_candidates(dataset_size: int) -> dict[str, Pipeline]:
    leaf_nodes = 15 if dataset_size < 5000 else 25
    min_leaf = 20 if dataset_size < 5000 else 35
    return {
        "ridge": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scale", StandardScaler()),
                ("model", Ridge(alpha=20.0)),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        learning_rate=0.05,
                        max_iter=220,
                        max_leaf_nodes=leaf_nodes,
                        min_samples_leaf=min_leaf,
                        l2_regularization=8.0,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def numeric_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    output = df.copy()
    for column in columns:
        if column not in output:
            output[column] = np.nan
        output[column] = pd.to_numeric(output[column], errors="coerce")
    return output


def add_game_derived(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    output["rest_difference"] = (
        pd.to_numeric(output["home_rest"], errors="coerce")
        - pd.to_numeric(output["away_rest"], errors="coerce")
    )
    output["abs_spread_line"] = pd.to_numeric(output["spread_line"], errors="coerce").abs()
    output["is_postseason"] = (output["game_type"].fillna("REG") != "REG").astype(int)
    return output


def add_prop_derived(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    output["rest_difference"] = (
        pd.to_numeric(output["team_rest"], errors="coerce")
        - pd.to_numeric(output["opponent_rest"], errors="coerce")
    )
    position = output["position"].fillna("").str.upper()
    for code in ["QB", "RB", "WR", "TE"]:
        output[f"pos_{code.lower()}"] = (position == code).astype(int)
    return output


def regression_metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    return {
        "rows": int(len(actual)),
        "mae": round(float(mean_absolute_error(actual, prediction)), 4),
        "rmse": round(float(math.sqrt(mean_squared_error(actual, prediction))), 4),
    }


def choose_model(
    df: pd.DataFrame,
    features: list[str],
    target_col: str,
    baseline_col: str,
) -> Selection:
    training = df[df["season"] < VALIDATION_SEASON].copy()
    validation = df[df["season"] == VALIDATION_SEASON].copy()
    if len(training) < 100 or len(validation) < 30:
        raise RuntimeError(
            f"Not enough rows for {target_col}: train={len(training)} validation={len(validation)}"
        )

    y_train = training[target_col].to_numpy(dtype=float)
    baseline_train = training[baseline_col].to_numpy(dtype=float)
    y_validation = validation[target_col].to_numpy(dtype=float)
    baseline_validation = validation[baseline_col].to_numpy(dtype=float)
    residual_train = y_train - baseline_train
    baseline_mae = float(mean_absolute_error(y_validation, baseline_validation))

    best: Selection | None = None
    for name, candidate in make_candidates(len(training)).items():
        model = clone(candidate)
        model.fit(training[features], residual_train)
        residual_prediction = model.predict(validation[features])
        for weight in BLEND_WEIGHTS:
            prediction = baseline_validation + weight * residual_prediction
            mae = float(mean_absolute_error(y_validation, prediction))
            if best is None or mae < best.validation_mae:
                best = Selection(
                    algorithm=name,
                    blend_weight=weight,
                    validation_mae=mae,
                    validation_baseline_mae=baseline_mae,
                    validation_predictions=prediction,
                )
    assert best is not None
    return best


def fit_selected(
    training: pd.DataFrame,
    features: list[str],
    target_col: str,
    baseline_col: str,
    selection: Selection,
) -> Pipeline:
    candidate = make_candidates(len(training))[selection.algorithm]
    model = clone(candidate)
    residual = (
        training[target_col].to_numpy(dtype=float)
        - training[baseline_col].to_numpy(dtype=float)
    )
    model.fit(training[features], residual)
    return model


def betting_metrics(
    actual: np.ndarray,
    market: np.ndarray,
    prediction: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    edge = prediction - market
    selected = np.abs(edge) >= threshold
    if not selected.any():
        return {
            "threshold": threshold,
            "bets": 0,
            "wins": 0,
            "losses": 0,
            "pushes": 0,
            "win_rate": 0.0,
            "roi": 0.0,
            "units": 0.0,
        }
    direction = np.sign(edge[selected])
    result = direction * (actual[selected] - market[selected])
    wins = int(np.sum(result > 0))
    losses = int(np.sum(result < 0))
    pushes = int(np.sum(result == 0))
    settled = wins + losses
    units = wins * (100.0 / 110.0) - losses
    bets = int(np.sum(selected))
    return {
        "threshold": threshold,
        "bets": bets,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "win_rate": round(wins / settled, 4) if settled else 0.0,
        "roi": round(units / bets, 4) if bets else 0.0,
        "units": round(units, 4),
    }


def choose_bet_threshold(
    actual: np.ndarray,
    market: np.ndarray,
    prediction: np.ndarray,
) -> tuple[float, list[dict[str, float | int]]]:
    evaluations = [
        betting_metrics(actual, market, prediction, threshold)
        for threshold in BET_THRESHOLDS
    ]
    eligible = [row for row in evaluations if int(row["bets"]) >= 40]
    if not eligible:
        eligible = [row for row in evaluations if int(row["bets"]) >= 20]
    if not eligible:
        eligible = evaluations
    selected = max(eligible, key=lambda row: (float(row["roi"]), int(row["bets"])))
    return float(selected["threshold"]), evaluations


def run_model_backtest(
    name: str,
    df: pd.DataFrame,
    features: list[str],
    target_col: str,
    baseline_col: str,
    output_dir: Path,
    evaluate_bets: bool = False,
) -> dict[str, Any]:
    required = list(dict.fromkeys(["season", target_col, baseline_col] + features))
    df = numeric_frame(df, required)
    df = df.dropna(subset=["season", target_col, baseline_col]).copy()
    df["season"] = df["season"].astype(int)

    selection = choose_model(df, features, target_col, baseline_col)
    validation = df[df["season"] == VALIDATION_SEASON].copy()
    validation_actual = validation[target_col].to_numpy(dtype=float)
    validation_baseline = validation[baseline_col].to_numpy(dtype=float)

    selected_threshold: float | None = None
    validation_bets: list[dict[str, float | int]] | None = None
    if evaluate_bets:
        selected_threshold, validation_bets = choose_bet_threshold(
            validation_actual,
            validation_baseline,
            selection.validation_predictions,
        )

    test_rows: list[pd.DataFrame] = []
    season_results: list[dict[str, Any]] = []
    for season in TEST_SEASONS:
        training = df[df["season"] < season].copy()
        testing = df[df["season"] == season].copy()
        if testing.empty:
            continue
        model = fit_selected(training, features, target_col, baseline_col, selection)
        residual_prediction = model.predict(testing[features])
        baseline = testing[baseline_col].to_numpy(dtype=float)
        prediction = baseline + selection.blend_weight * residual_prediction
        actual = testing[target_col].to_numpy(dtype=float)

        testing = testing.copy()
        testing["prediction"] = prediction
        testing["baseline_prediction"] = baseline
        test_rows.append(testing)

        result: dict[str, Any] = {
            "season": season,
            "model": regression_metrics(actual, prediction),
            "baseline": regression_metrics(actual, baseline),
        }
        result["mae_improvement_pct"] = round(
            100.0 * (result["baseline"]["mae"] - result["model"]["mae"])
            / result["baseline"]["mae"],
            3,
        )
        if evaluate_bets and selected_threshold is not None:
            result["betting"] = betting_metrics(
                actual,
                baseline,
                prediction,
                selected_threshold,
            )
        season_results.append(result)

    if not test_rows:
        raise RuntimeError(f"No unseen test rows available for {name}")
    combined = pd.concat(test_rows, ignore_index=True)
    combined_actual = combined[target_col].to_numpy(dtype=float)
    combined_prediction = combined["prediction"].to_numpy(dtype=float)
    combined_baseline = combined["baseline_prediction"].to_numpy(dtype=float)
    model_metrics = regression_metrics(combined_actual, combined_prediction)
    baseline_metrics = regression_metrics(combined_actual, combined_baseline)

    final_model = fit_selected(df, features, target_col, baseline_col, selection)
    artifact_path = output_dir / f"{name}.joblib"
    joblib.dump(
        {
            "model_version": MODEL_VERSION,
            "name": name,
            "estimator": final_model,
            "features": features,
            "target": target_col,
            "baseline": baseline_col,
            "blend_weight": selection.blend_weight,
            "selected_bet_threshold": selected_threshold,
            "trained_through_season": int(df["season"].max()),
        },
        artifact_path,
    )

    report: dict[str, Any] = {
        "selected_algorithm": selection.algorithm,
        "blend_weight": selection.blend_weight,
        "approved_vs_baseline_on_validation": (
            selection.validation_mae < selection.validation_baseline_mae
        ),
        "validation": {
            "season": VALIDATION_SEASON,
            "rows": int(len(validation)),
            "model_mae": round(selection.validation_mae, 4),
            "baseline_mae": round(selection.validation_baseline_mae, 4),
        },
        "test": {
            "seasons": TEST_SEASONS,
            "model": model_metrics,
            "baseline": baseline_metrics,
            "mae_improvement_pct": round(
                100.0 * (baseline_metrics["mae"] - model_metrics["mae"])
                / baseline_metrics["mae"],
                3,
            ),
            "by_season": season_results,
        },
        "artifact": artifact_path.name,
    }
    if evaluate_bets and selected_threshold is not None:
        report["betting"] = {
            "odds_assumption": "-110 flat stake",
            "threshold_selected_on_validation": selected_threshold,
            "validation_threshold_grid": validation_bets,
            "unseen_test": betting_metrics(
                combined_actual,
                combined_baseline,
                combined_prediction,
                selected_threshold,
            ),
        }
    return report


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# NFL Baseline Model Backtest",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "Chronological design: train through 2022, select on 2023, and test on unseen 2024-2025 games.",
        "",
        "| Model | Algorithm | Test rows | Model MAE | Baseline MAE | Improvement |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, model in report["models"].items():
        test = model["test"]
        lines.append(
            f"| {name} | {model['selected_algorithm']} | {test['model']['rows']} | "
            f"{test['model']['mae']:.3f} | {test['baseline']['mae']:.3f} | "
            f"{test['mae_improvement_pct']:.2f}% |"
        )
    lines.extend(["", "## Market-line betting tests", ""])
    for name in ["game_margin", "game_total"]:
        model = report["models"][name]
        betting = model.get("betting", {}).get("unseen_test", {})
        lines.append(
            f"- **{name}:** threshold {model.get('betting', {}).get('threshold_selected_on_validation')} points; "
            f"{betting.get('bets', 0)} bets; {100 * betting.get('win_rate', 0):.1f}% win rate; "
            f"{100 * betting.get('roi', 0):.1f}% ROI at assumed -110 odds."
        )
    lines.extend(
        [
            "",
            "Player yardage models are evaluated against the player's prior five-game average. "
            "Historical sportsbook player-prop lines are not yet stored, so player-prop ROI is not claimed.",
            "",
        ]
    )
    return "\n".join(lines)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Run chronological NFL baseline model backtests")
    parser.add_argument("--output-dir", default="model-output")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    supabase = SupabaseClient()
    run_id: int | None = None
    try:
        run_id = supabase.start_run()

        game_df = supabase.fetch_all(
            "matchup_training_features",
            GAME_SELECT,
            filters={
                "season": "gte.2016",
                "actual_home_margin": "not.is.null",
                "actual_total": "not.is.null",
                "spread_line": "not.is.null",
                "total_line": "not.is.null",
                "home_prior_games": "gte.3",
                "away_prior_games": "gte.3",
            },
            order="season.asc,week.asc,game_id.asc",
        )
        game_df = add_game_derived(game_df)

        report: dict[str, Any] = {
            "model_version": MODEL_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "design": {
                "train_seasons": "2016-2022",
                "validation_season": VALIDATION_SEASON,
                "unseen_test_seasons": TEST_SEASONS,
                "future_information_leakage": "rolling features exclude the current game",
            },
            "models": {},
        }

        report["models"]["game_margin"] = run_model_backtest(
            "game_margin",
            game_df,
            GAME_FEATURES,
            "actual_home_margin",
            "spread_line",
            output_dir,
            evaluate_bets=True,
        )
        report["models"]["game_total"] = run_model_backtest(
            "game_total",
            game_df,
            GAME_FEATURES,
            "actual_total",
            "total_line",
            output_dir,
            evaluate_bets=True,
        )

        for name, config in PROP_CONFIGS.items():
            columns = COMMON_PROP_SELECT + config["select"]
            prop_df = supabase.fetch_all(
                "player_prop_training_features",
                columns,
                filters={"season": "gte.2016", **config["filters"]},
                order="season.asc,week.asc,game_id.asc,player_id.asc",
            )
            prop_df = add_prop_derived(prop_df)
            report["models"][name] = run_model_backtest(
                name,
                prop_df,
                config["features"],
                config["target"],
                config["baseline"],
                output_dir,
                evaluate_bets=False,
            )

        safe_report = json_safe(report)
        (output_dir / "backtest_report.json").write_text(
            json.dumps(safe_report, indent=2, sort_keys=True), encoding="utf-8"
        )
        (output_dir / "backtest_summary.md").write_text(
            build_markdown(safe_report), encoding="utf-8"
        )
        manifest = {
            "model_version": MODEL_VERSION,
            "trained_through_season": 2025,
            "models": {
                name: {
                    "artifact": model["artifact"],
                    "selected_algorithm": model["selected_algorithm"],
                    "blend_weight": model["blend_weight"],
                    "approved_vs_baseline_on_validation": model[
                        "approved_vs_baseline_on_validation"
                    ],
                }
                for name, model in safe_report["models"].items()
            },
        }
        (output_dir / "model_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )

        supabase.finish_run(run_id, "success", safe_report)
        print(build_markdown(safe_report))
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        if run_id is not None:
            try:
                supabase.finish_run(run_id, "failed", error=error[:2000])
            except Exception:
                traceback.print_exc()
        raise
    finally:
        supabase.close()


if __name__ == "__main__":
    main()
