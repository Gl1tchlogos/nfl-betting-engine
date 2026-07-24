            if len(subset) < 20:
                continue
            rows.append({
                "slice_dimension": dimension,
                "slice_value": str(value),
                "rows": int(len(subset)),
                "mae": float(subset["abs_error"].mean()),
                "bias": float(subset["error"].mean()),
                "rmse": float(np.sqrt(np.mean(np.square(subset["error"])))),
                "interval_80_coverage": float(subset["covered"].mean()),
            })
    return rows


def game_prediction_rows(
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
            "home_team": values.get("home_team"),
            "away_team": values.get("away_team"),
            "is_indoor": values.get("is_indoor"),
            "high_wind": values.get("high_wind"),
            "freezing_temperature": values.get("freezing_temperature"),
            "absolute_spread": values.get("absolute_spread"),
            "total_line": values.get("total_line"),
            "retrained_this_week": values.get("retrained_this_week"),
        }
        rows.append({
            "run_id": run_id,
            "model_version": MODEL_VERSION,
            "prediction_mode": "locked_first_pass",
            "model_name": target_name,
            "algorithm": selected.name,
            "game_id": values["game_id"],
            "player_id": None,
            "season": int(values["season"]),
            "week": int(values["week"]),
            "team": values.get("home_team"),
            "opponent": values.get("away_team"),
            "position_group": "GAME",
            "target_name": target,
            "projected_mean": safe_float(values.get("prediction")),
            "projected_median": safe_float(values.get("prediction")),
            "projected_sd": safe_float(values.get("projection_sd")),
            "lower_80": safe_float(values.get("lower_80")),
            "upper_80": safe_float(values.get("upper_80")),
            "actual_value": safe_float(values.get(target)),
            "baseline_value": safe_float(values.get(baseline)),
            "feature_as_of": json_safe(values.get("kickoff_time")),
            "training_rows": int(values.get("training_rows", 0)),
            "metadata": json_safe(metadata),
        })
    return rows

def run_game_target(
    store: Store,
    run_id: int,
    target_name: str,
    frame: pd.DataFrame,
    retrain_every: int,
    output_dir: Path,
) -> dict[str, Any]:
    config = GAME_TARGETS[target_name]
    prepared, features = prepare_game_target(frame, config)
    selected, candidates, validation = choose_model(
        prepared, features, config["target"], config["baseline"]
    )
    calibration = calibration_from_validation(
        validation.assign(position_group="GAME"), config["target"]
    )
    predictions = walk_forward(
        prepared.assign(position_group="GAME"),
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
    ablations = run_ablation(
        prepared.assign(position_group="GAME"),
        selected,
        config["groups"],
        features,
        config["target"],
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
        game_prediction_rows(
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
            for row in build_game_error_slices(predictions, config["target"])
        ],
    )
    final_model = fit_model(selected, prepared.assign(position_group="GAME"), features, config["target"])
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run comprehensive NFL walk-forward models")
    parser.add_argument("--output-dir", default="comprehensive-model-output")
    parser.add_argument("--retrain-every", type=int, default=1)
    parser.add_argument("--skip-game-models", action="store_true")
    args = parser.parse_args()
    if args.retrain_every < 1:
        raise ValueError("--retrain-every must be at least 1")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    store = Store()
    run_id: int | None = None
    try:
        configuration = {
            "development_through": DEVELOPMENT_END_SEASON,
            "validation_season": VALIDATION_SEASON,
            "test_seasons": TEST_SEASONS,
            "retrain_every_weeks": args.retrain_every,
            "prediction_mode": "locked_first_pass",
            "feature_time_rule": "all rolling and expanding features exclude the current game",
        }
        run_id = store.start_walkforward(configuration)
        store.upsert_rows("model_feature_catalog", catalog_rows())

        summary: dict[str, Any] = {
            "model_version": MODEL_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "configuration": configuration,
            "models": {},
        }

        domains = {config["domain"] for config in PLAYER_TARGETS.values()}
        domain_frames: dict[str, pd.DataFrame] = {}
        for domain in sorted(domains):
            columns = select_columns_for_domain(domain)
            frame = store.fetch_all(
                "player_comprehensive_model_features",
                columns,
                filters={"season": "gte.2016"},
                order="season.asc,week.asc,game_id.asc,player_id.asc",
            )
            target_columns = [
                config["target"] for config in PLAYER_TARGETS.values() if config["domain"] == domain
            ]
            domain_frames[domain] = add_player_derived(frame, target_columns)

        for target_name, config in PLAYER_TARGETS.items():
            print(f"Running {target_name}...")
            summary["models"][target_name] = run_player_target(
                store,
                run_id,
                target_name,
                domain_frames[config["domain"]],
                args.retrain_every,
                output_dir,
            )

        if not args.skip_game_models:
            game_derived = {"is_postseason", "late_season"}
            game_columns = list(
                dict.fromkeys(
                    GAME_ID_COLUMNS
                    + [
                        feature
                        for config in GAME_TARGETS.values()
                        for feature in all_features(config["groups"])
                        if feature not in game_derived
                    ]
                    + ["home_prior_games", "away_prior_games"]
                )
            )
            game_frame = store.fetch_all(
                "game_comprehensive_model_features",
                game_columns,
                filters={"season": "gte.2016"},
                order="season.asc,week.asc,game_id.asc",
            )
            game_frame = add_game_derived(game_frame)
            for target_name in GAME_TARGETS:
                print(f"Running {target_name}...")
                summary["models"][target_name] = run_game_target(
                    store,
                    run_id,
                    target_name,
                    game_frame,
                    args.retrain_every,
                    output_dir,
                )

        safe_summary = json_safe(summary)
        (output_dir / "comprehensive_report.json").write_text(
            json.dumps(safe_summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        markdown = report_markdown(safe_summary)
        (output_dir / "comprehensive_summary.md").write_text(markdown, encoding="utf-8")
        manifest = {
            "model_version": MODEL_VERSION,
            "models": {
                name: {
                    "artifact": f"{name}.joblib",
                    "selected_algorithm": result["selected_algorithm"],
                    "walk_forward_mae": result["walk_forward"]["mae"],
                    "baseline_mae": result["baseline"]["mae"],
                    "improvement_pct": result["improvement_pct"],
                    "projection_only": name in PLAYER_TARGETS,
                }
                for name, result in safe_summary["models"].items()
            },
        }
        (output_dir / "model_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        store.finish_walkforward(run_id, "success", safe_summary)
        print(markdown)
    except Exception as exc:
        traceback.print_exc()
        if run_id is not None:
            try:
                store.finish_walkforward(
                    run_id,
                    "failed",
                    error=f"{type(exc).__name__}: {exc}"[:2000],
                )
            except Exception:
                traceback.print_exc()
        raise
    finally:
        store.close()


if __name__ == "__main__":
    main()
