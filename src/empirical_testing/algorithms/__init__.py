"""Learning algorithms: fixed-rate Algorithm 1, its span-adaptive sibling, and the NLHF baselines."""

from empirical_testing.algorithms.one_expert import (
    FixedRates,
    GroupwiseSelfPlayPolicy,
    fit_groupwise_self_play,
    fixed_rates,
    kl_divergence,
)
from empirical_testing.algorithms.nlhf import NLHFPolicy, fit_pooled_nlhf
from empirical_testing.algorithms.span_adaptive import (
    FrozenRates,
    GroupwiseSpanAdaptivePolicy,
    SpanAdaptiveSchedule,
    fit_groupwise_span_adaptive,
)
from empirical_testing.algorithms.registry import (
    FIXED_ID,
    ONLINE_ALGORITHMS,
    SPAN_ADAPTIVE_ID,
    fit_online_learner,
    is_adaptive,
    rate_diagnostics,
)

__all__ = [
    "FIXED_ID",
    "FixedRates",
    "FrozenRates",
    "GroupwiseSelfPlayPolicy",
    "GroupwiseSpanAdaptivePolicy",
    "NLHFPolicy",
    "ONLINE_ALGORITHMS",
    "SPAN_ADAPTIVE_ID",
    "SpanAdaptiveSchedule",
    "fit_groupwise_self_play",
    "fit_groupwise_span_adaptive",
    "fit_online_learner",
    "fit_pooled_nlhf",
    "fixed_rates",
    "is_adaptive",
    "kl_divergence",
    "rate_diagnostics",
]
