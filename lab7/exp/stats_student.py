"""lab7 Part A: the statistics of a performance claim. Fill in TODO(S1, S2, S3).

    python -m unittest lab7.tests.test_stats -v

Performance numbers on a laptop GPU are noisy (clock boost, temperature,
background processes). We therefore summarise repeated runs with the MEDIAN
and attach a bootstrap confidence interval, and we decide a hypothesis by
comparing that interval with a PRE-REGISTERED minimal effect. Lecture L7 §3.
Pure Python (random, statistics); no numpy needed.
"""

import random
import statistics


# ------------------------------------------------------------------ S1 ----
# TODO(S1) percentile bootstrap confidence interval of a statistic.
#   repeat n_boot times: draw len(xs) values from xs WITH replacement
#   (rng.choices), compute stat on the draw; sort the n_boot results and
#   return (lo, hi) = the alpha/2 and 1 - alpha/2 empirical quantiles,
#   i.e. results[int(alpha/2 * n_boot)] and results[int((1 - alpha/2) * n_boot) - 1].
# Use rng = random.Random(seed) so results are reproducible.
def bootstrap_ci(xs, stat=statistics.median, n_boot=10000, alpha=0.05, seed=0):
    if len(xs) == 0:
        raise ValueError("bootstrap_ci of an empty sample")
    raise NotImplementedError("TODO S1 bootstrap_ci")


# ------------------------------------------------------------------ S2 ----
# TODO(S2) effect of a treatment as a RATIO of medians, treatment / baseline,
# with a bootstrap CI that resamples BOTH groups independently on each draw:
#   point = median(treat) / median(base)
#   each draw: b = rng.choices(base, k=len(base)), t = rng.choices(treat, k=len(treat));
#              ratio = median(t) / median(b)
#   (lo, hi) as in S1.
# Return (point, lo, hi). A ratio of 0.8 for a latency means "20% faster".
def ratio_ci(base, treat, n_boot=10000, alpha=0.05, seed=0):
    if not base or not treat:
        raise ValueError("ratio_ci needs both samples")
    raise NotImplementedError("TODO S2 ratio_ci")


# ------------------------------------------------------------------ S3 ----
# TODO(S3) turn a ratio CI into a verdict against a pre-registered margin.
#   direction "lower":  the treatment should make the metric smaller by at
#                       least min_effect (e.g. 0.10 = 10%): target = 1 - min_effect
#       "supported"    if hi <  target   (confidently at least that much better)
#       "refuted"      if lo >  target   (confidently less than that, or worse)
#       "inconclusive" otherwise
#   direction "higher": target = 1 + min_effect
#       "supported" if lo > target; "refuted" if hi < target; else "inconclusive"
# Anything else -> ValueError. "Refuted" is a RESULT, not a failure: it tells
# you the effect is smaller than what you said would matter.
def decide(lo, hi, direction, min_effect):
    raise NotImplementedError("TODO S3 decide")
