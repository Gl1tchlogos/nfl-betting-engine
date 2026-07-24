
# Apply domain-specific eligibility filters before downloading player rows.
# This reduces transfer volume without changing the population used by the model.
_store_fetch_all = Store.fetch_all


def _optimized_fetch_all(
    self: Store,
    table: str,
    columns: list[str],
    filters: dict[str, str] | None = None,
    order: str = "season.asc,week.asc",
) -> pd.DataFrame:
    optimized_filters = dict(filters or {})
    if table == "player_comprehensive_model_features":
        optimized_filters.setdefault("prior_games", "gte.3")
        if "passing_attempts_avg_3" in columns:
            optimized_filters.setdefault("passing_attempts_avg_3", "gte.10")
        elif "carries_avg_3" in columns:
            optimized_filters.setdefault("carries_avg_3", "gte.4")
        elif "targets_avg_3" in columns:
            optimized_filters.setdefault("targets_avg_3", "gte.3")
    return _store_fetch_all(
        self,
        table,
        columns,
        filters=optimized_filters,
        order=order,
    )


Store.fetch_all = _optimized_fetch_all
