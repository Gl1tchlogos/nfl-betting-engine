# Comprehensive Walk-Forward Run 4 Results

Generated: 2026-07-24
Model version: `comprehensive_v1`
Official database run: `model_walkforward_runs.id = 4`
Status: Success
Runtime: approximately 14 minutes

## Test design

- Development seasons: 2016-2022
- Model selection season: 2023
- Locked weekly walk-forward seasons: 2024-2025
- Models retrained as the simulation advanced through the test weeks.
- Rolling, expanding, and exponentially weighted features excluded the game being predicted.
- The first stored prediction for each historical player-game or game was treated as the locked result.

## Stored outputs

- 23,335 locked walk-forward predictions
- 50 candidate-model evaluations
- 54 feature-family ablation tests
- 670 diagnostic error slices
- Seven player projection models approved
- Two game models rejected
- Zero models approved for betting before price-aware historical line testing

## Combined 2024-2025 performance

| Model | Selected algorithm | Rows | Model MAE | Baseline MAE | Improvement | Central 80% coverage |
|---|---|---:|---:|---:|---:|---:|
| Passing attempts | Histogram quantile median | 1,213 | 7.485 | 8.641 | 13.380% | 80.21% |
| Passing yards | Ridge/boosting ensemble | 1,213 | 62.910 | 71.395 | 11.884% | 84.58% |
| Carries | Position-specific histogram quantile median | 2,497 | 3.637 | 3.854 | 5.653% | 78.89% |
| Rushing yards | Position-specific histogram quantile median | 2,497 | 22.478 | 23.573 | 4.645% | 77.17% |
| Targets | Histogram quantile median | 4,925 | 2.115 | 2.219 | 4.689% | 80.04% |
| Receptions | Ridge/boosting ensemble | 4,925 | 1.657 | 1.721 | 3.713% | 79.88% |
| Receiving yards | Histogram quantile median | 4,925 | 22.433 | 23.751 | 5.549% | 80.89% |
| Game margin | Ridge/boosting ensemble | 570 | 9.792 | 9.687 | -1.085% | 82.63% |
| Game total | Ridge/boosting ensemble | 570 | 10.269 | 10.096 | -1.719% | 77.54% |

## Season consistency

Every approved player model beat its baseline separately in both unseen seasons.

| Model | 2024 improvement | 2025 improvement |
|---|---:|---:|
| Passing attempts | 11.796% | 14.977% |
| Passing yards | 10.705% | 13.055% |
| Carries | 5.568% | 5.742% |
| Rushing yards | 3.707% | 5.614% |
| Targets | 4.714% | 4.662% |
| Receptions | 3.574% | 3.859% |
| Receiving yards | 5.223% | 5.893% |
| Game margin | -0.885% | -1.286% |
| Game total | -1.835% | -1.610% |

## Approval status

Projection approved:

- Passing attempts
- Passing yards
- Carries
- Rushing yards
- Targets
- Receptions
- Receiving yards

Rejected:

- Game margin
- Game total

Betting approval remains false for every model. Historical player-prop lines and prices are required before player models can be graded as betting systems. Game models must pass a separate price-aware strategy test before betting approval.

## Feature-family findings

A positive number below means validation error increased when that family was removed, indicating that the family contributed useful information.

Strongest observed contributions included:

- Game total market/environment: +4.722% MAE when removed
- Passing-attempt continuity: +3.548%
- Passing-yard continuity: +2.885%
- Game-margin market/environment: +2.589%
- Passing-yard environment: +2.179%
- Carries opportunity: +1.686%
- Passing-attempt matchup: +1.599%
- Carries continuity: +1.393%
- Rushing-yard opportunity: +1.370%
- Receptions trend: +1.295%
- Targets trend: +1.184%

Potentially noisy groups that should be isolated in later experiments included:

- Passing-yard efficiency: removing it improved validation MAE by 0.767%
- Target environment: removing it improved validation MAE by 0.475%
- Receiving-yard matchup: removing it improved validation MAE by 0.338%
- Passing-attempt efficiency: removing it improved validation MAE by 0.330%
- Game-total matchup: removing it improved validation MAE by 0.185%

These are research hypotheses, not automatic feature-removal decisions. Any adjustment must be tested on later untouched data.

## Largest diagnostic weaknesses

Relative to each model's average MAE:

- High-usage rushing-yard rows were 35.3% harder.
- High-usage carries were 27.2% harder.
- High-usage receiving-yard rows were 25.9% harder.
- Quarterback passing attempts after at least 14 days away were 20.8% harder.
- High-usage receptions were 20.1% harder.
- Low-volume quarterback passing yards were 18.0% harder.
- High-usage targets were 18.5% harder.
- Freezing-temperature passing attempts were 12.5% harder.
- Quarterback passing yards after at least 14 days away were 10.1% harder.
- High-wind rushing yards were 7.5% harder.

These findings support later controlled experiments for workload ceilings, return-from-absence handling, and weather-specific adjustments.

## Data quality

Final quality run: `data_quality_runs.id = 3`
Status: Success

All monitored issue counts were zero, including:

- Orphan player and team statistics
- Duplicate player-game and team-game keys
- Duplicate comprehensive player and game feature keys
- Base player and team feature leakage
- Advanced player and team feature leakage
- Defense-by-position feature leakage
- Successful walk-forward runs without predictions
- Predictions without a registered model
- Unresolved prop players and prop quotes without a game

## Operational additions completed with this run

- Comprehensive player, team, defense-by-position, and game feature layers
- Role and opportunity features
- Advanced efficiency and explosive-play features
- Opponent-adjusted position features
- Exponentially weighted recency
- Sample-size-shrunk baselines
- Uncertainty intervals
- Weekly locked walk-forward storage
- Candidate-model comparison
- Feature ablation
- Error slicing
- Automatic projection approval gates
- Guardrail tests for chronology and direct-target leakage
- Clean handling of unpublished current-season NFLverse weekly files
