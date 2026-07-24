    output["receptions_per_target_avg_5"] = output["receptions_avg_5"] / output[
        "targets_avg_5"
    ].replace(0, np.nan)
    output["position_passing_yards_per_attempt_allowed_avg_5"] = output[
        "position_passing_yards_allowed_avg_5"
    ] / output["position_passing_attempts_allowed_avg_5"].replace(0, np.nan)
    output["position_rushing_yards_per_carry_allowed_avg_5"] = output[
        "position_rushing_yards_allowed_avg_5"
    ] / output["position_carries_allowed_avg_5"].replace(0, np.nan)
    output["position_receiving_yards_per_target_allowed_avg_5"] = output[
        "position_receiving_yards_allowed_avg_5"
    ] / output["position_targets_allowed_avg_5"].replace(0, np.nan)

    output = add_week_position_priors(output, targets)
    base_map = {
        "passing_attempts": "passing_attempts_avg_5",
        "passing_yards": "passing_yards_avg_5",
        "carries": "carries_avg_5",
        "rushing_yards": "rushing_yards_avg_5",
        "targets": "targets_avg_5",
        "receptions": "receptions_avg_5",
        "receiving_yards": "receiving_yards_avg_5",
    }
    for target in targets:
        player_average = pd.to_numeric(output[base_map[target]], errors="coerce")
        position_average = pd.to_numeric(output[f"{target}_position_prior"], errors="coerce")
        weight = pd.to_numeric(output["prior_games"], errors="coerce").fillna(0)
        weight = weight / (weight + POSITION_SHRINKAGE_GAMES)
        output[f"{target}_shrunk_baseline"] = (
            weight * player_average + (1.0 - weight) * position_average
        ).fillna(player_average).fillna(position_average)
    return output


def add_game_derived(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for column in output.columns:
        if column not in GAME_ID_COLUMNS and column not in {"home_team", "away_team", "game_type"}:
            output[column] = pd.to_numeric(output[column], errors="coerce")
    output["is_postseason"] = (~output["game_type"].astype(str).str.upper().isin(["REG", "REGULAR"])).astype(int)
    output["late_season"] = (pd.to_numeric(output["week"], errors="coerce") >= 13).astype(int)
    return output


def select_columns_for_domain(domain: str) -> list[str]:
    configs = [config for config in PLAYER_TARGETS.values() if config["domain"] == domain]
    columns: list[str] = []
    base = [
        "game_id", "player_id", "full_name", "position", "position_group",
        "season", "week", "season_type", "team", "opponent", "game_date",
        "game_ts", "is_home", "prior_games", "birth_date", "years_experience",
        "height_inches", "weight_lbs", "days_since_previous_game", "roof",
        "surface", "temperature", "wind", "team_spread", "total_line",
        "implied_team_points", "implied_opponent_points", "team_rest",
        "opponent_rest", "changed_team", "return_after_14_days", "is_indoor",
        "high_wind", "freezing_temperature", "passing_attempt_share",
        "carry_share", "target_share",
    ]
    for column in base:
        if column not in columns:
            columns.append(column)
    for config in configs:
        for column in [config["target"], config["baseline"].replace("_shrunk_baseline", "_avg_5")]:
            if column not in columns:
                columns.append(column)
        for feature in all_features(config["groups"]):
            source_feature = feature
            if feature.endswith("_ewm_5") or feature.endswith("_shrunk_baseline"):
                continue
            if feature in {
                "age_years", "rest_difference", "surface_grass", "is_postseason",
                "late_season", "team_favored", "passing_yards_per_attempt_avg_5",
                "rushing_yards_per_carry_avg_5", "receiving_yards_per_target_avg_5",
                "receptions_per_target_avg_5",
                "position_passing_yards_per_attempt_allowed_avg_5",
                "position_rushing_yards_per_carry_allowed_avg_5",
                "position_receiving_yards_per_target_allowed_avg_5",
            }:
                continue
            if source_feature not in columns:
                columns.append(source_feature)
    return columns


def prepare_target(frame: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    target = config["target"]
    baseline = config["baseline"]
    features = [feature for feature in all_features(config["groups"]) if feature in frame.columns]
    eligible_column, threshold = config["eligibility"]
    output = frame.copy()
    output[target] = pd.to_numeric(output[target], errors="coerce")
    output[baseline] = pd.to_numeric(output[baseline], errors="coerce")
    output = output[
        (pd.to_numeric(output["prior_games"], errors="coerce") >= 3)
        & (pd.to_numeric(output[eligible_column], errors="coerce") >= threshold)
    ].dropna(subset=[target, baseline, "season", "week"])
    output["season"] = output["season"].astype(int)
    output["week"] = output["week"].astype(int)
    return output.reset_index(drop=True), features


def candidate_predictions(
    spec: ModelSpec,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
    target: str,
) -> np.ndarray:
    fitted = fit_model(spec, train, features, target)
    return predict_model(spec, fitted, validation, features)


def choose_model(
    frame: pd.DataFrame,
    features: list[str],
    target: str,
    baseline: str,
) -> tuple[ModelSpec, list[dict[str, Any]], pd.DataFrame]:
    train = frame[frame["season"] <= DEVELOPMENT_END_SEASON].copy()
    validation = frame[frame["season"] == VALIDATION_SEASON].copy()
    if len(train) < MIN_TRAIN_ROWS or validation.empty:
        raise RuntimeError(f"Insufficient development or validation rows for {target}")

    candidates = [
        ModelSpec("ridge"),
        ModelSpec("hist_gradient_boosting"),
        ModelSpec("hist_quantile_median"),
        ModelSpec("ensemble"),
    ]
    rows: list[dict[str, Any]] = []
    prediction_map: dict[str, np.ndarray] = {}
    actual = validation[target].to_numpy(dtype=float)
    baseline_prediction = validation[baseline].to_numpy(dtype=float)
    baseline_metrics = regression_metrics(actual, baseline_prediction)
    rows.append({"algorithm": "shrunk_baseline", **baseline_metrics})

    for spec in candidates:
        prediction = candidate_predictions(spec, train, validation, features, target)
        prediction_map[spec.name] = prediction
        rows.append({"algorithm": spec.name, **regression_metrics(actual, prediction)})

    best_global = min(
        [row for row in rows if row["algorithm"] != "shrunk_baseline"],
        key=lambda row: float(row["mae"]),
    )
    best_algorithm = str(best_global["algorithm"])
    if best_algorithm == "ensemble":
        base_algorithm = "ensemble"
    else:
        base_algorithm = best_algorithm
    position_spec = ModelSpec(base_algorithm, position_specific=True)
    if "position_group" in validation.columns and validation["position_group"].nunique(dropna=False) > 1:
        eligible_positions = validation.groupby("position_group", dropna=False).size()
        if (eligible_positions >= MIN_POSITION_VALIDATION_ROWS).any():
            prediction = candidate_predictions(position_spec, train, validation, features, target)
            prediction_map[position_spec.name] = prediction
            rows.append({"algorithm": position_spec.name, **regression_metrics(actual, prediction)})

    selected_row = min(
        [row for row in rows if row["algorithm"] != "shrunk_baseline"],
        key=lambda row: float(row["mae"]),
    )
    selected_name = str(selected_row["algorithm"])
    selected = ModelSpec(
        selected_name.replace("position_", ""),
        position_specific=selected_name.startswith("position_"),
    )
    validation = validation.copy()
    validation["prediction"] = prediction_map[selected_name]
    validation["baseline_prediction"] = baseline_prediction
    return selected, rows, validation


def calibration_from_validation(validation: pd.DataFrame, target: str) -> dict[str, Any]:
    residual = validation[target].to_numpy(dtype=float) - validation["prediction"].to_numpy(dtype=float)
    result: dict[str, Any] = {
        "global_sd": max(float(np.std(residual, ddof=1)), 0.01),
        "global_abs_q80": max(float(np.quantile(np.abs(residual), 0.80)), 0.01),
        "positions": {},
    }
    for position, subset in validation.groupby("position_group", dropna=False):
        if len(subset) < 75:
            continue
        values = subset[target].to_numpy(dtype=float) - subset["prediction"].to_numpy(dtype=float)
        result["positions"][str(position)] = {
            "sd": max(float(np.std(values, ddof=1)), 0.01),
            "abs_q80": max(float(np.quantile(np.abs(values), 0.80)), 0.01),
        }
    return result


def apply_intervals(frame: pd.DataFrame, prediction: np.ndarray, calibration: dict[str, Any]) -> pd.DataFrame:
    output = frame.copy()
    output["prediction"] = prediction
    sd = np.full(len(output), calibration["global_sd"], dtype=float)
    half_width = np.full(len(output), calibration["global_abs_q80"], dtype=float)
    if "position_group" in output.columns:
        for position, indexes in output.groupby("position_group", dropna=False).groups.items():
            local = calibration["positions"].get(str(position))
            if local:
                positions = output.index.get_indexer(indexes)
                sd[positions] = local["sd"]
                half_width[positions] = local["abs_q80"]
    output["projection_sd"] = sd
    output["lower_80"] = prediction - half_width
    output["upper_80"] = prediction + half_width
    return output


def run_ablation(
    frame: pd.DataFrame,
    selected: ModelSpec,
    groups: dict[str, list[str]],
    features: list[str],
    target: str,
) -> list[dict[str, Any]]:
    train = frame[frame["season"] <= DEVELOPMENT_END_SEASON]
    validation = frame[frame["season"] == VALIDATION_SEASON]
    full_prediction = candidate_predictions(selected, train, validation, features, target)
    full_mae = float(mean_absolute_error(validation[target], full_prediction))
    results: list[dict[str, Any]] = []
    for group_name, group_features in groups.items():
        reduced = [feature for feature in features if feature not in group_features]
        if not reduced:
            continue
        prediction = candidate_predictions(selected, train, validation, reduced, target)
        ablated_mae = float(mean_absolute_error(validation[target], prediction))
        change = ablated_mae - full_mae
        results.append(
            {
                "feature_group_removed": group_name,
                "rows": int(len(validation)),
                "baseline_mae": full_mae,
                "ablated_mae": ablated_mae,
                "mae_change": change,
                "mae_change_pct": 100.0 * change / full_mae if full_mae else 0.0,
            }
        )
    return results


def walk_forward(
    frame: pd.DataFrame,
    selected: ModelSpec,
    features: list[str],
    target: str,
    baseline: str,
    calibration: dict[str, Any],
    retrain_every: int,
) -> pd.DataFrame:
    test = frame[frame["season"].isin(TEST_SEASONS)].copy()
    periods = (
        test[["season", "week"]].drop_duplicates().sort_values(["season", "week"]).itertuples(index=False)
    )
    output: list[pd.DataFrame] = []
    fitted: Any = None
    fitted_at = -1
    period_number = 0
    for season, week in periods:
        period_number += 1
        current = test[(test["season"] == season) & (test["week"] == week)].copy()
        training = frame[(frame["season"] < season) | ((frame["season"] == season) & (frame["week"] < week))].copy()
        if len(training) < MIN_TRAIN_ROWS or current.empty:
            continue
        if fitted is None or (period_number - fitted_at) >= retrain_every:
            fitted = fit_model(selected, training, features, target)
            fitted_at = period_number
        prediction = predict_model(selected, fitted, current, features)
        current = apply_intervals(current, prediction, calibration)
        current["baseline_prediction"] = current[baseline].to_numpy(dtype=float)
        current["training_rows"] = len(training)
        current["retrained_this_week"] = period_number == fitted_at
        output.append(current)
    if not output:
        raise RuntimeError(f"No walk-forward predictions generated for {target}")
    return pd.concat(output, ignore_index=True)


def usage_tier(frame: pd.DataFrame, target_name: str) -> pd.Series:
    if target_name in {"passing_attempts", "passing_yards"}:
        source = pd.to_numeric(frame["passing_attempts_avg_3"], errors="coerce")
        return pd.cut(source, [-np.inf, 20, 30, np.inf], labels=["low", "medium", "high"]).astype(str)
    if target_name in {"carries", "rushing_yards"}:
        source = pd.to_numeric(frame["carries_avg_3"], errors="coerce")
        return pd.cut(source, [-np.inf, 8, 15, np.inf], labels=["low", "medium", "high"]).astype(str)
    source = pd.to_numeric(frame["targets_avg_3"], errors="coerce")
    return pd.cut(source, [-np.inf, 5, 8, np.inf], labels=["low", "medium", "high"]).astype(str)


def build_error_slices(frame: pd.DataFrame, target: str, target_name: str) -> list[dict[str, Any]]:
    data = frame.copy()
    data["error"] = data["prediction"] - data[target]
    data["abs_error"] = data["error"].abs()
    data["covered"] = ((data[target] >= data["lower_80"]) & (data[target] <= data["upper_80"])).astype(int)
    data["usage_tier"] = usage_tier(data, target_name)
    uncertainty_rank = pd.to_numeric(data["projection_sd"], errors="coerce").rank(
        method="average", pct=True
    )
    data["confidence_tier"] = pd.cut(
        uncertainty_rank,
