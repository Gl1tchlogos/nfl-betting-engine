from __future__ import annotations

import argparse
import json
import math
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from models.backtest_baselines import SupabaseClient
from models.comprehensive_config import (
    DEVELOPMENT_END_SEASON,
    GAME_ID_COLUMNS,
    GAME_TARGETS,
    MIN_POSITION_TRAIN_ROWS,
    MIN_POSITION_VALIDATION_ROWS,
    MIN_TRAIN_ROWS,
    MODEL_VERSION,
    PLAYER_TARGETS,
    TEST_SEASONS,
    VALIDATION_SEASON,
)

PAGE_SIZE = 500
POSITION_SHRINKAGE_GAMES = 8.0
CENTRAL_80_Z = 1.2815515655446004


@dataclass(frozen=True)
class ModelSpec:
    algorithm: str
    position_specific: bool = False

    @property
    def name(self) -> str:
        return f"position_{self.algorithm}" if self.position_specific else self.algorithm


class Store(SupabaseClient):
    def start_walkforward(self, configuration: dict[str, Any]) -> int:
        response = self._check(
            self.client.post(
                "/model_walkforward_runs",
                json={
                    "model_version": MODEL_VERSION,
                    "run_type": "locked_walkforward",
                    "status": "running",
                    "configuration": configuration,
                },
                headers={"Prefer": "return=representation"},
            )
        )
        return int(response.json()[0]["id"])

    def finish_walkforward(
        self,
        run_id: int,
        status: str,
        summary: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        self._check(
            self.client.patch(
                "/model_walkforward_runs",
                params={"id": f"eq.{run_id}"},
                json={
                    "status": status,
                    "summary": summary,
                    "error_message": error,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                },
                headers={"Prefer": "return=minimal"},
            )
        )

    def insert_rows(self, table: str, rows: list[dict[str, Any]], batch_size: int = 400) -> None:
        if not rows:
            return
        for start in range(0, len(rows), batch_size):
            self._check(
                self.client.post(
                    f"/{table}",
                    json=rows[start : start + batch_size],
                    headers={"Prefer": "return=minimal"},
                )
            )

    def upsert_rows(self, table: str, rows: list[dict[str, Any]], batch_size: int = 400) -> None:
        if not rows:
            return
        for start in range(0, len(rows), batch_size):
            self._check(
                self.client.post(
                    f"/{table}",
                    json=rows[start : start + batch_size],
                    headers={"Prefer": "resolution=merge-duplicates,return=minimal"},
                )
            )


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if not math.isfinite(number) else number


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return safe_float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def all_features(groups: dict[str, list[str]]) -> list[str]:
    output: list[str] = []
    for columns in groups.values():
        for column in columns:
            if column not in output:
                output.append(column)
    return output


def make_estimator(algorithm: str, rows: int) -> Any:
    if algorithm == "ridge":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scale", StandardScaler()),
                ("model", Ridge(alpha=18.0)),
            ]
        )
    if algorithm in {"hist_gradient_boosting", "hist_quantile_median"}:
        kwargs: dict[str, Any] = {
            "learning_rate": 0.045,
            "max_iter": 180,
            "max_leaf_nodes": 20 if rows < 5000 else 28,
            "min_samples_leaf": 20 if rows < 5000 else 35,
            "l2_regularization": 2.0,
            "early_stopping": True,
            "random_state": 20260724,
        }
        if algorithm == "hist_quantile_median":
            kwargs.update({"loss": "quantile", "quantile": 0.5})
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("model", HistGradientBoostingRegressor(**kwargs)),
            ]
        )
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def fit_model(spec: ModelSpec, train: pd.DataFrame, features: list[str], target: str) -> Any:
    if spec.algorithm == "ensemble":
        return {
            "ridge": fit_model(ModelSpec("ridge", spec.position_specific), train, features, target),
            "hist": fit_model(
                ModelSpec("hist_gradient_boosting", spec.position_specific), train, features, target
            ),
        }
    if spec.position_specific:
        global_model = make_estimator(spec.algorithm, len(train))
        global_model.fit(train[features], train[target])
        models: dict[str, Any] = {}
        for position, subset in train.groupby("position_group", dropna=False):
            if len(subset) < MIN_POSITION_TRAIN_ROWS:
                continue
            model = make_estimator(spec.algorithm, len(subset))
            model.fit(subset[features], subset[target])
            models[str(position)] = model
        return {"global": global_model, "positions": models}
    model = make_estimator(spec.algorithm, len(train))
    model.fit(train[features], train[target])
    return model


def predict_model(spec: ModelSpec, fitted: Any, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    if spec.algorithm == "ensemble":
        ridge = predict_model(ModelSpec("ridge", spec.position_specific), fitted["ridge"], frame, features)
        hist = predict_model(
            ModelSpec("hist_gradient_boosting", spec.position_specific), fitted["hist"], frame, features
        )
        return (ridge + hist) / 2.0
    if spec.position_specific:
        result = np.asarray(fitted["global"].predict(frame[features]), dtype=float)
        for position, indexes in frame.groupby("position_group", dropna=False).groups.items():
            model = fitted["positions"].get(str(position))
            if model is not None:
                positions = frame.index.get_indexer(indexes)
                result[positions] = model.predict(frame.loc[indexes, features])
        return result
    return np.asarray(fitted.predict(frame[features]), dtype=float)


def regression_metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    return {
        "rows": int(len(actual)),
        "mae": round(float(mean_absolute_error(actual, prediction)), 5),
        "rmse": round(float(mean_squared_error(actual, prediction) ** 0.5), 5),
        "bias": round(float(np.mean(prediction - actual)), 5),
    }


def add_week_position_priors(frame: pd.DataFrame, targets: Iterable[str]) -> pd.DataFrame:
    output = frame.copy()
    ordered_keys = ["season", "week"]
    for target in targets:
        grouped = (
            output.groupby(["position_group", *ordered_keys], dropna=False)[target]
            .agg(["sum", "count"])
            .reset_index()
            .sort_values(["position_group", *ordered_keys])
        )
        grouped["prior_sum"] = grouped.groupby("position_group", dropna=False)["sum"].cumsum() - grouped["sum"]
        grouped["prior_count"] = grouped.groupby("position_group", dropna=False)["count"].cumsum() - grouped["count"]
        grouped[f"{target}_position_prior"] = grouped["prior_sum"] / grouped["prior_count"].replace(0, np.nan)
        output = output.merge(
            grouped[["position_group", *ordered_keys, f"{target}_position_prior"]],
            on=["position_group", *ordered_keys],
            how="left",
        )
    return output


def add_player_derived(frame: pd.DataFrame, targets: Iterable[str]) -> pd.DataFrame:
    output = frame.copy()
    output["game_date"] = pd.to_datetime(output["game_date"], errors="coerce")
    output["game_ts"] = pd.to_datetime(output["game_ts"], errors="coerce", utc=True)
    output["birth_date"] = pd.to_datetime(output["birth_date"], errors="coerce")
    output["age_years"] = (output["game_date"] - output["birth_date"]).dt.days / 365.25
    output["rest_difference"] = pd.to_numeric(output["team_rest"], errors="coerce") - pd.to_numeric(
        output["opponent_rest"], errors="coerce"
    )
    output["surface_grass"] = output["surface"].astype(str).str.lower().str.contains("grass").astype(int)
    output["is_postseason"] = (~output["season_type"].astype(str).str.upper().isin(["REG", "REGULAR"])).astype(int)
    output["late_season"] = (pd.to_numeric(output["week"], errors="coerce") >= 13).astype(int)
    output["team_favored"] = (pd.to_numeric(output["team_spread"], errors="coerce") > 0).astype(int)

    numeric_candidates = set(all_features({k: v for config in PLAYER_TARGETS.values() for k, v in config["groups"].items()}))
    numeric_candidates.update(targets)
    numeric_candidates.update(
        {
            "passing_attempt_share", "carry_share", "target_share",
            "passing_attempts", "passing_yards", "carries", "rushing_yards",
            "targets", "receptions", "receiving_yards",
        }
    )
    for column in numeric_candidates:
        if column in output.columns:
            output[column] = pd.to_numeric(output[column], errors="coerce")

    output = output.sort_values(["player_id", "game_ts", "game_id"]).reset_index(drop=True)
    ewm_columns = [
        "passing_attempts", "passing_yards", "passing_attempt_share",
        "carries", "rushing_yards", "carry_share", "targets", "receptions",
        "receiving_yards", "target_share",
    ]
    for column in ewm_columns:
        if column not in output.columns:
            continue
        output[f"{column}_ewm_5"] = output.groupby("player_id", sort=False)[column].transform(
            lambda series: series.shift(1).ewm(span=5, adjust=False, min_periods=2).mean()
        )

    output["passing_yards_per_attempt_avg_5"] = output["passing_yards_avg_5"] / output[
        "passing_attempts_avg_5"
    ].replace(0, np.nan)
    output["rushing_yards_per_carry_avg_5"] = output["rushing_yards_avg_5"] / output[
        "carries_avg_5"
    ].replace(0, np.nan)
    output["receiving_yards_per_target_avg_5"] = output["receiving_yards_avg_5"] / output[
        "targets_avg_5"
    ].replace(0, np.nan)
