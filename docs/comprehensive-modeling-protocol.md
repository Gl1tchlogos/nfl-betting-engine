# Comprehensive NFL Modeling Protocol

## Purpose

Build player and game projections using only information available before kickoff, then evaluate them through locked weekly walk-forward testing. Historical sportsbook player-prop lines are deliberately excluded until the projection layer has been validated independently.

## Chronological design

- Development data: 2016-2022
- Model and feature-family selection: 2023
- Locked weekly walk-forward test: 2024-2025
- Each test week is predicted using only prior games.
- The model may retrain after a week is revealed, but the first stored prediction for that week is the only result that counts as simulated live performance.

## Projection targets

### Player volume

- Passing attempts
- Carries
- Targets
- Receptions

### Player yardage

- Passing yards
- Rushing yards
- Receiving yards

### Game markets

- Home point margin
- Total points

## Feature families

### Trend

- Prior 3-, 5-, and 10-game averages
- Exponentially weighted recent averages
- Short-term versus longer-term trend changes
- Prior-game standard deviation
- Sample-size-shrunk player baselines

### Opportunity and role

- Carry share
- Target share
- Air-yard share
- Reception share
- Receiving-yard share
- Team pass and rush volume
- Team change indicator
- Return after a long absence

### Efficiency

- EPA
- CPOE
- Air yards
- Yards after catch
- First downs
- Explosive plays
- Yards per attempt, carry, and target
- Fumbles and sacks

### Opponent matchup

- Defense-by-position volume and yardage allowed
- Defensive sacks and quarterback hits
- Interceptions and passes defended
- Tackles for loss and forced fumbles
- Explosive plays allowed

### Game environment

- Spread and total
- Implied team and opponent points
- Home/away
- Rest difference
- Regular season versus postseason
- Indoor/outdoor, wind, temperature, and surface context

Weather-specific historical splits and coaching history remain a separately tracked adjustment backlog. They should be promoted only when later error analysis demonstrates value.

## Candidate models

- Shrunk historical baseline
- Ridge regression
- Histogram gradient boosting
- Median quantile gradient boosting
- Ridge/boosting ensemble
- Position-specific versions when sample sizes are sufficient

The selected model is the candidate with the lowest 2023 mean absolute error. Complexity is not rewarded unless it improves validation performance.

## Uncertainty

Each projection includes:

- Point estimate
- Estimated standard deviation
- Central 80% interval
- Position-specific residual calibration when the validation sample is sufficient

Coverage is measured during the locked 2024-2025 walk-forward test.

## Ablation testing

After selecting a model, each feature family is removed individually and the model is refit on the development period. The change in 2023 validation error is recorded. This identifies feature families that contribute measurable value and those that only add complexity.

## Error analysis

Player errors are sliced by:

- Season
- Position
- Team and opponent
- Home/away
- Usage level
- Confidence tier
- Season phase
- Indoor/outdoor
- High wind and freezing temperature
- Team change
- Return after absence

Game errors are sliced by:

- Season phase
- Spread bucket
- Total bucket
- Indoor/outdoor
- High wind and freezing temperature
- Home and away team

## Approval rules

A model is projection-approved only when:

1. It has at least 200 locked test predictions.
2. It beats its baseline across the combined unseen test.
3. It beats its baseline separately in both 2024 and 2025.
4. All required test seasons are present.

No player model may be betting-approved until real historical prop lines and prices are loaded and a separate price-aware ROI test passes. Game models also remain betting-disabled until a dedicated strategy test evaluates edge thresholds, actual prices, drawdown, and season-to-season stability.

## Stored outputs

- Complete run configuration and status
- Locked first-pass weekly predictions
- Candidate model metrics
- Feature ablation results
- Error slices
- Fitted model artifacts
- JSON and Markdown reports
- Model registry approval status
- Data-quality snapshot
