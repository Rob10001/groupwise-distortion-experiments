"""The four policy families the maintained empirical suite may execute.

This module is the **single allowlist** consulted at every validation and
execution boundary: configuration loading, dry-run inventories, training
dispatch and figure inputs.  A method that is not named here cannot be fitted
by a maintained runner, whatever a configuration file asks for.

The comparison suite compares exactly four policy families:

=====================  ==================  ==========================================
Display name           Internal identity   Training
=====================  ==================  ==========================================
Multigroup GLHF         ``multigroup``      the span-adaptive learner, per-group
Pooled GLHF            ``pooled``          the same learner with pooled memberships
NLHF                   ``pooled_nlhf``     pooled, passive-trained batch solver
Uniform reference      ``uniform``         not fitted at all
=====================  ==================  ==========================================

Three families that earlier expanded-suite revisions declared -- intersection
NLHF, single-group specialists and specialist mixtures -- are **retired**.
Their names remain in :data:`RETIRED_METHODS` so obsolete configurations are
rejected explicitly. No maintained runner may execute them.

Two distinctions this module deliberately does *not* encode:

* **Sampling protocol is not identity.**  ``active_multigroup`` and
  ``multigroup`` are the same policy family trained under different sampling
  protocols, so they canonicalize to the same name.  The protocol travels
  beside the method, never inside it -- that is what lets an active and a
  passive cell be compared as the same method.
* **Uniform is not a fitter.**  It is in the allowlist because it is a reported
  method, and excluded from :data:`FITTED_METHODS` because nothing ever fits
  it.  Evaluation-only welfare optimization is a separate matter and remains
  allowed: it defines the distortion metric rather than producing a policy.
"""

from __future__ import annotations

from typing import Iterable, Mapping

#: Internal identity -> display name, in reporting order.
ALLOWED_METHODS: Mapping[str, str] = {
    "multigroup": "Multigroup GLHF",
    "pooled": "Pooled GLHF",
    "pooled_nlhf": "NLHF",
    "uniform": "Uniform reference",
}

#: The subset that a fitter is ever invoked for.  ``uniform`` is absent by
#: construction: it has no parameters to learn, and a maintained runner that
#: tries to fit it is a bug, not a configuration choice.
FITTED_METHODS: tuple[str, ...] = ("multigroup", "pooled", "pooled_nlhf")

#: Historical spellings that name one of the four families.  The sampling
#: protocol these names imply is carried separately and is not lost here.
METHOD_ALIASES: Mapping[str, str] = {
    "active_multigroup": "multigroup",
    "active_pooled": "pooled",
    "uniform_reference": "uniform",
    "nlhf": "pooled_nlhf",
}

#: Families removed from maintained empirical execution.  Listed explicitly so
#: a configuration naming one fails with a specific message instead of the
#: generic "unknown method", and so the acceptance tests have something to
#: assert against.
RETIRED_METHODS: Mapping[str, str] = {
    "intersection_nlhf": "intersection NLHF",
    "specialist_mixture": "specialist mixtures",
    "specialist": "single-group specialists",
}

#: Configuration wildcard that used to admit one specialist per group.
RETIRED_METHOD_PREFIXES: tuple[str, ...] = ("specialist:",)


class RetiredMethodError(ValueError):
    """A retired policy family was named where execution could reach it."""


class UnknownMethodError(ValueError):
    """A method outside the allowlist was named."""


def is_retired(method: str) -> bool:
    """Whether ``method`` names a retired family, wildcard spellings included."""

    name = str(method)
    return name in RETIRED_METHODS or name.startswith(RETIRED_METHOD_PREFIXES)


def canonical(method: str) -> str:
    """The internal identity ``method`` names.

    Raises :class:`RetiredMethodError` for a retired family and
    :class:`UnknownMethodError` for anything else outside the allowlist, so a
    caller never has to distinguish "removed on purpose" from "typo".
    """

    name = str(method)
    if is_retired(name):
        retired = RETIRED_METHODS.get(name) or RETIRED_METHODS["specialist"]
        raise RetiredMethodError(
            f"{name!r} names {retired}, which is retired from maintained empirical "
            f"execution; the permitted methods are {sorted(ALLOWED_METHODS)}"
        )
    name = METHOD_ALIASES.get(name, name)
    if name not in ALLOWED_METHODS:
        raise UnknownMethodError(
            f"unknown method {method!r}; the permitted methods are {sorted(ALLOWED_METHODS)}"
        )
    return name


def is_allowed(method: str) -> bool:
    """Whether ``method`` names one of the four permitted families."""

    try:
        canonical(method)
    except (RetiredMethodError, UnknownMethodError):
        return False
    return True


def require_allowed(methods: Iterable[str], *, where: str) -> tuple[str, ...]:
    """Canonicalize ``methods``, refusing retired and unknown names.

    ``where`` names the boundary being guarded and is quoted in the error, so a
    failure says which configuration key, stage or figure input was at fault.
    """

    resolved: list[str] = []
    for method in methods:
        try:
            resolved.append(canonical(method))
        except (RetiredMethodError, UnknownMethodError) as error:
            raise type(error)(f"{where}: {error}") from None
    return tuple(resolved)


def require_fittable(methods: Iterable[str], *, where: str) -> tuple[str, ...]:
    """Canonicalize ``methods`` and refuse any that must never reach a fitter.

    The allowlist check comes first, so a retired family is still reported as
    retired rather than as "not fittable".
    """

    resolved = require_allowed(methods, where=where)
    for method in resolved:
        if method not in FITTED_METHODS:
            raise ValueError(
                f"{where}: {method!r} is never fitted "
                f"({ALLOWED_METHODS[method]} has no parameters to learn)"
            )
    return resolved
