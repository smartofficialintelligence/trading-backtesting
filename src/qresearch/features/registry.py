"""Name -> implementation registries for features and transforms.

Configuration files name a ``kind`` and pass ``params``; the registry resolves the class
and coerces parameters through each dataclass's own annotations (so ``"PT5M"`` becomes a
``timedelta`` and a list becomes a tuple) without the class knowing about YAML.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any, Final

from pydantic import TypeAdapter

from qresearch.features.contracts import Feature
from qresearch.features.session import (
    IsEarlyClose,
    MinutesSinceOpen,
    MinutesToClose,
    SessionFraction,
)
from qresearch.features.technical import (
    BarRange,
    DayOfWeek,
    LaggedReturn,
    MinuteOfDay,
    RelativeVolume,
    RollingVolatility,
)
from qresearch.features.transforms import Transform, Winsorizer, ZScoreScaler

FEATURES: Final[dict[str, type[Any]]] = {
    "lagged_return": LaggedReturn,
    "rolling_volatility": RollingVolatility,
    "relative_volume": RelativeVolume,
    "bar_range": BarRange,
    "minute_of_day": MinuteOfDay,
    "day_of_week": DayOfWeek,
    "minutes_since_open": MinutesSinceOpen,
    "minutes_to_close": MinutesToClose,
    "session_fraction": SessionFraction,
    "is_early_close": IsEarlyClose,
}

TRANSFORMS: Final[dict[str, type[Any]]] = {
    "zscore": ZScoreScaler,
    "winsorize": Winsorizer,
}


def construct(cls: type[Any], params: Mapping[str, Any]) -> Any:
    """Instantiate a dataclass, coercing each parameter through its field annotation."""
    if not dataclasses.is_dataclass(cls):
        return cls(**params)
    fields = {f.name: f for f in dataclasses.fields(cls) if f.init}
    unknown = sorted(set(params) - set(fields))
    if unknown:
        raise ValueError(
            f"{cls.__name__}: unknown parameters {unknown}; accepted: {sorted(fields)}"
        )
    coerced = {}
    for name, value in params.items():
        annotation = fields[name].type
        if isinstance(annotation, str):
            # ``from __future__ import annotations`` leaves strings; resolve via the module.
            import sys

            module = sys.modules[cls.__module__]
            annotation = eval(annotation, vars(module))
        coerced[name] = TypeAdapter(annotation).validate_python(value, strict=False)
    return cls(**coerced)


def build_feature(kind: str, params: Mapping[str, Any]) -> Feature:
    try:
        cls = FEATURES[kind]
    except KeyError:
        raise KeyError(f"unknown feature kind {kind!r}; known: {sorted(FEATURES)}") from None
    feature = construct(cls, params)
    assert isinstance(feature, Feature)
    return feature


def build_transform(kind: str, columns: tuple[str, ...], params: Mapping[str, Any]) -> Transform:
    try:
        cls = TRANSFORMS[kind]
    except KeyError:
        raise KeyError(f"unknown transform kind {kind!r}; known: {sorted(TRANSFORMS)}") from None
    transform = construct(cls, {"columns": tuple(columns), **params})
    return transform  # type: ignore[no-any-return]
