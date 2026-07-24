        [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0],
        labels=["high", "medium", "low"],
        include_lowest=True,
    ).astype(str)
    data["week_phase"] = pd.cut(
        data["week"], [0, 4, 8, 12, 99], labels=["early", "first_half", "second_half", "late"]
    ).astype(str)
    dimensions = [
        "season", "position_group", "team", "opponent", "is_home",
        "usage_tier", "confidence_tier", "week_phase", "is_indoor",
        "high_wind", "freezing_temperature", "changed_team",
        "return_after_14_days",
    ]
    rows: list[dict[str, Any]] = []
    for dimension in dimensions:
        if dimension not in data.columns:
            continue
        for value, subset in data.groupby(dimension, dropna=False):
            if len(subset) < 20:
                continue
            rows.append(
                {
                    "slice_dimension": dimension,
                    "slice_value": str(value),
                    "rows": int(len(subset)),
                    "mae": float(subset["abs_error"].mean()),
                    "bias": float(subset["error"].mean()),
                    "rmse": float(np.sqrt(np.mean(np.square(subset["error"])))),
                    "interval_80_coverage": float(subset["covered"].mean()),
                }
            )
    return rows


def prediction_rows(
    run_id: int,
    target_name: str,
    selected: ModelSpec,
    frame: pd.DataFrame,
    target: str,
    baseline: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        values = row._asdict()
        metadata = {
            "full_name": values.get("full_name"),
            "is_home": values.get("is_home"),
            "usage_tier": str(usage_tier(pd.DataFrame([values]), target_name).iloc[0]),
            "position_group": values.get("position_group"),
            "team": values.get("team"),
            "opponent": values.get("opponent"),
            "is_indoor": values.get("is_indoor"),
            "high_wind": values.get("high_wind"),
            "freezing_temperature": values.get("freezing_temperature"),
            "changed_team": values.get("changed_team"),
            "return_after_14_days": values.get("return_after_14_days"),
            "retrained_this_week": values.get("retrained_this_week"),
        }
        rows.append(
            {
                "run_id": run_id,
                "model_version": MODEL_VERSION,
                "prediction_mode": "locked_first_pass",
                "model_name": target_name,
                "algorithm": selected.name,
                "game_id": values["game_id"],
                "player_id": values.get("player_id"),
                "season": int(values["season"]),
                "week": int(values["week"]),
                "team": values.get("team"),
                "opponent": values.get("opponent"),
                "position_group": values.get("position_group"),
                "target_name": target,
                "projected_mean": safe_float(values.get("prediction")),
                "projected_median": safe_float(values.get("prediction")),
                "projected_sd": safe_float(values.get("projection_sd")),
                "lower_80": safe_float(values.get("lower_80")),
                "upper_80": safe_float(values.get("upper_80")),
                "actual_value": safe_float(values.get(target)),
                "baseline_value": safe_float(values.get(baseline)),
                "feature_as_of": json_safe(values.get("game_ts")),
                "training_rows": int(values.get("training_rows", 0)),
                "metadata": json_safe(metadata),
            }
        )
    return rows


def candidate_metric_rows(
    run_id: int,
    model_name: str,
    target: str,
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "run_id": run_id,
            "model_name": model_name,
            "target_name": target,
            "algorithm": str(row["algorithm"]),
            "evaluation_period": str(VALIDATION_SEASON),
            "rows": int(row["rows"]),
            "mae": safe_float(row.get("mae")),
            "rmse": safe_float(row.get("rmse")),
            "metadata": {"bias": safe_float(row.get("bias"))},
        }
        for row in candidates
    ]


def catalog_rows() -> list[dict[str, Any]]:
    catalog: dict[str, dict[str, str]] = {}
    for model_name, config in {**PLAYER_TARGETS, **GAME_TARGETS}.items():
        for group, features in config["groups"].items():
            for feature in features:
                catalog.setdefault(
                    feature,
                    {
                        "feature_name": feature,
                        "feature_group": group,
                        "entity_level": "player" if model_name in PLAYER_TARGETS else "game",
                        "description": feature.replace("_", " "),
                        "leakage_rule": "Computed only from information available before the predicted game.",
                        "active": True,
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
    return list(catalog.values())


def report_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Comprehensive NFL Walk-Forward Report",
        "",
        f"Generated: {summary['generated_at']}",
        "",
        "Development through 2022, model selection on 2023, locked walk-forward testing on 2024-2025.",
        "",
        "| Model | Selected | Rows | MAE | Baseline MAE | Improvement | 80% coverage |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for name, result in summary["models"].items():
        lines.append(
            f"| {name} | {result['selected_algorithm']} | {result['walk_forward']['rows']} | "
            f"{result['walk_forward']['mae']:.3f} | {result['baseline']['mae']:.3f} | "
            f"{result['improvement_pct']:.2f}% | {100 * result['interval_80_coverage']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "Player projections use opportunity, efficiency, matchup, game environment, continuity, "
            "exponentially weighted recency, and sample-size shrinkage features.",
            "",
            "Player-prop betting ROI remains disabled until historical lines and prices are loaded.",
        ]
    )
    return "\n".join(lines)


def run_player_target(
    store: Store,
    run_id: int,
    target_name: str,
    frame: pd.DataFrame,
    retrain_every: int,
    output_dir: Path,
) -> dict[str, Any]:
    config = PLAYER_TARGETS[target_name]
    prepared, features = prepare_target(frame, config)
    selected, candidates, validation = choose_model(
        prepared, features, config["target"], config["baseline"]
    )
    calibration = calibration_from_validation(validation, config["target"])
    ablations = run_ablation(
        prepared, selected, config["groups"], features, config["target"]
    )
    predictions = walk_forward(
        prepared,
        selected,
        features,
        config["target"],
        config["baseline"],
        calibration,
        retrain_every,
    )
    actual = predictions[config["target"]].to_numpy(dtype=float)
    point = predictions["prediction"].to_numpy(dtype=float)
    baseline_prediction = predictions["baseline_prediction"].to_numpy(dtype=float)
    metrics = regression_metrics(actual, point)
    baseline_metrics = regression_metrics(actual, baseline_prediction)
    coverage = float(
        np.mean((actual >= predictions["lower_80"].to_numpy()) & (actual <= predictions["upper_80"].to_numpy()))
    )

    store.insert_rows(
        "model_candidate_metrics",
        candidate_metric_rows(run_id, target_name, config["target"], candidates),
    )
    store.insert_rows(
        "model_ablation_results",
        [
            {
                "run_id": run_id,
                "model_name": target_name,
                "target_name": config["target"],
                "algorithm": selected.name,
                **json_safe(row),
            }
            for row in ablations
        ],
    )
    store.insert_rows(
        "model_walkforward_predictions",
        prediction_rows(
            run_id, target_name, selected, predictions, config["target"], config["baseline"]
        ),
    )
    store.insert_rows(
        "model_error_slices",
        [
            {
                "run_id": run_id,
                "model_name": target_name,
                "target_name": config["target"],
                "prediction_mode": "locked_first_pass",
                **json_safe(row),
            }
            for row in build_error_slices(predictions, config["target"], target_name)
        ],
    )

    final_model = fit_model(selected, prepared, features, config["target"])
    joblib.dump(
        {
            "model_version": MODEL_VERSION,
            "model_name": target_name,
            "selected_spec": selected,
            "features": features,
            "target": config["target"],
            "baseline": config["baseline"],
            "calibration": calibration,
            "estimator": final_model,
            "trained_through_season": int(prepared["season"].max()),
        },
        output_dir / f"{target_name}.joblib",
    )
    return {
        "selected_algorithm": selected.name,
        "candidate_metrics": candidates,
        "walk_forward": metrics,
        "baseline": baseline_metrics,
        "improvement_pct": 100.0 * (baseline_metrics["mae"] - metrics["mae"]) / baseline_metrics["mae"],
        "interval_80_coverage": coverage,
        "interval_mean_width": float((predictions["upper_80"] - predictions["lower_80"]).mean()),
        "ablation": ablations,
    }


def prepare_game_target(frame: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    output = frame.copy()
    features = [feature for feature in all_features(config["groups"]) if feature in output.columns]
    output[config["target"]] = pd.to_numeric(output[config["target"]], errors="coerce")
    output[config["baseline"]] = pd.to_numeric(output[config["baseline"]], errors="coerce")
    output = output.dropna(subset=[config["target"], config["baseline"], "season", "week"])
    output = output[(output["home_prior_games"] >= 3) & (output["away_prior_games"] >= 3)]
    output["season"] = output["season"].astype(int)
    output["week"] = output["week"].astype(int)
    return output.reset_index(drop=True), features



def build_game_error_slices(frame: pd.DataFrame, target: str) -> list[dict[str, Any]]:
    data = frame.copy()
    data["error"] = data["prediction"] - data[target]
    data["abs_error"] = data["error"].abs()
    data["covered"] = ((data[target] >= data["lower_80"]) & (data[target] <= data["upper_80"])).astype(int)
    data["week_phase"] = pd.cut(
        data["week"], [0, 4, 8, 12, 99], labels=["early", "first_half", "second_half", "late"]
    ).astype(str)
    data["spread_bucket"] = pd.cut(
        pd.to_numeric(data["absolute_spread"], errors="coerce"),
        [-np.inf, 3, 7, np.inf],
        labels=["close", "moderate", "large"],
    ).astype(str)
    data["total_bucket"] = pd.cut(
        pd.to_numeric(data["total_line"], errors="coerce"),
        [-np.inf, 42, 48, np.inf],
        labels=["low", "medium", "high"],
    ).astype(str)
    dimensions = [
        "season", "week_phase", "is_indoor", "high_wind",
        "freezing_temperature", "spread_bucket", "total_bucket",
        "home_team", "away_team",
    ]
    rows: list[dict[str, Any]] = []
    for dimension in dimensions:
        if dimension not in data.columns:
            continue
        for value, subset in data.groupby(dimension, dropna=False):
