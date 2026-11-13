# Risk readout: night operations in Region 3

**Question.** Can the fleet expand night operations (time blocks 22,23,0,1,2,3,4) into Region 3 while holding the critical-event rate at or below the assurance threshold of 2.0 events per 1,000 miles with at least 95% posterior probability?

## Method in three sentences

Raw geo-tagged event and mileage logs were rolled up with Spark SQL into a 12 x 12 cell by 24 hour-of-day exposure table covering 60 days and 1.20 million fleet miles. A Bayesian hierarchical Poisson model with a log-exposure offset, a spatial ICAR effect, an AR(1) hour-of-day effect, partial region pooling, and a cell x time-block nugget was fitted in PyMC. The assurance decision integrates the posterior over the cells and night blocks in the requested region, exposure-weighting each draw, and asks for the posterior probability that the rate is at or below the threshold.

## Result

- Posterior rate: **0.85 events per 1,000 miles** (95% CrI 0.70 to 1.03).
- P(rate <= 2.0) = **1.000** (gate: 95%).
- Decision: **GO** for Region 3 (region_id=3), time blocks 22, 23, 0, 1, 2, 3, 4 (night).
- Precision: No additional mileage is required to reach the target CrI width of 0.50; the current width is 0.33 events per 1k miles.

## Comparison with the pooled fleet-wide baseline

The pooled Poisson GLM has a point estimate of 1.30 events per 1k miles for this slice and would say GO on a point-estimate basis. For the headline question the pooled point estimate agrees (it implies GO; the Bayesian rule returns GO). Across all 18 region x time-slice decisions, the pooled model declares GO in 15, whereas the Bayesian model returns GO in 12, NO-GO in 4, and INSUFFICIENT EVIDENCE in 2; the two rules disagree in 3 slices. The disagreements are concentrated in Region 4, where the pooled covariate-only rate sits below the threshold but the regionally resolved posterior interval does not clear it.

| Region | Slice | Posterior mean | 95% CrI | P(rate > 2.0) | Decision |
|---:|---|---:|---|---:|---|
| 5 | evening | 2.61 | [2.25, 2.99] | 1.000 | NO-GO |
| 4 | evening | 2.52 | [2.18, 2.90] | 0.999 | NO-GO |
| 5 | day | 2.37 | [2.09, 2.66] | 0.996 | NO-GO |

## Key assumptions

- Exposure (fleet miles) is measured exactly and enters as a known offset.
- The event rate is log-linear in two standardized covariates plus additive spatial, temporal, region, and nugget effects; the rate is constant within a cell x hour-of-day over the 60-day window.
- The modelled outcome is all recorded safety events (all severities). Because critical events are a subset (observed critical rate 0.079 per 1k miles), the all-event gate is stricter than a critical-only gate and is used conservatively.

## Top three caveats

1. **Simulated data.** The ground truth is a seeded simulation, so the excellent recovery (RMSE 0.47 vs pooled GLM 1.09; 95% CrI empirical coverage 95.7%) validates the pipeline, not real safety performance.
2. **Spatial smoothing and identifiability.** The ICAR prior shrinks local effects toward neighbours (posterior-mean phi correlation 0.73 with truth), so a narrow hotspot smaller than a few cells can be attenuated even when its regional rate is high.
3. **Unmodelled uncertainty.** Exposure error, severity misclassification, model-form error, and day-level heterogeneity beyond the nugget are not in the interval. The decision should be treated as a lower bound on uncertainty.

## Validation context (from this build)

- 95% credible-interval empirical coverage across all 3,456 cell x time-block rates: 95.7%.
- LOO improvement of the full model over the pooled Bayesian baseline: 188.8 elpd (difference SE 24.6).
- Prior-sensitivity sweep: maximum shift in the fleet-wide rate estimate 0.010 events per 1k miles.

Figures: `figures/risk_map.png`, `figures/spatial_recovery.png`, `figures/temporal_recovery.png`, `figures/ppc_stats.png`.

_Generated programmatically from artifacts in this repository; the numbers above are not hand-entered._
