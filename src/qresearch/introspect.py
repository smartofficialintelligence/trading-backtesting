"""Describe registered components so a form can be generated from them.

The registries map a ``kind`` to an implementation; the implementations are dataclasses
with annotated fields and defaults. That is enough to render a parameter form, which means
**adding a feature in code makes it appear in the UI with no UI change** — the same
property that keeps the YAML path and the browser path from drifting.

Only the shapes the registries actually use are described. A parameter whose type is not
recognised is reported as ``unsupported`` rather than guessed at, so it shows up as a
gap instead of silently rendering the wrong control.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import sys
import typing
from collections.abc import Mapping
from types import NoneType, UnionType
from typing import Any, Literal, Union, get_args, get_origin

from pydantic import Field

from qresearch.config import FrozenModel
from qresearch.features.registry import FEATURES, TRANSFORMS
from qresearch.strategy.registry import STRATEGIES

ParamType = Literal[
    "int", "float", "str", "bool", "choice", "duration", "mapping", "list", "unsupported"
]


class ParamSpec(FrozenModel):
    """One constructor parameter, in terms a form control can be built from."""

    name: str
    type: ParamType
    required: bool
    default: Any | None = None
    choices: tuple[str, ...] = ()
    optional: bool = False
    """True when the annotation admits None, so the control may be left blank."""

    annotation: str = ""
    """The original annotation, shown for parameters typed ``unsupported``."""


class ComponentSpec(FrozenModel):
    """A registered feature, transform, or strategy."""

    kind: str
    implementation: str
    summary: str = ""
    params: tuple[ParamSpec, ...] = ()

    @property
    def required_params(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.params if p.required)


def _summary(cls: type[Any]) -> str:
    """First line of the docstring — enough to label a form field."""
    doc = (cls.__doc__ or "").strip()
    return doc.split("\n", 1)[0] if doc else ""


def _describe(annotation: Any) -> tuple[ParamType, tuple[str, ...], bool]:
    """Map an annotation to (control type, choices, optional)."""
    optional = False
    origin = get_origin(annotation)
    if origin is Union or origin is UnionType:
        args = [a for a in get_args(annotation) if a is not NoneType]
        optional = len(args) < len(get_args(annotation))
        if len(args) != 1:
            return "unsupported", (), optional
        annotation = args[0]
        origin = get_origin(annotation)

    if origin is Literal:
        return "choice", tuple(str(a) for a in get_args(annotation)), optional
    if annotation is bool:
        return "bool", (), optional
    if annotation is int:
        return "int", (), optional
    if annotation is float:
        return "float", (), optional
    if annotation is str:
        return "str", (), optional
    if annotation is _dt.timedelta:
        return "duration", (), optional
    if origin in (tuple, list) or annotation in (tuple, list):
        return "list", (), optional
    if origin in (dict, Mapping) or annotation in (dict, Mapping):
        return "mapping", (), optional
    return "unsupported", (), optional


def describe(kind: str, cls: type[Any]) -> ComponentSpec:
    """Describe one registered implementation."""
    implementation = f"{cls.__module__}.{cls.__qualname__}"
    if not dataclasses.is_dataclass(cls):
        return ComponentSpec(kind=kind, implementation=implementation, summary=_summary(cls))

    module = sys.modules.get(cls.__module__)
    hints = typing.get_type_hints(cls, vars(module) if module else None)
    params = []
    for field in dataclasses.fields(cls):
        if not field.init:
            continue
        annotation = hints.get(field.name, field.type)
        control, choices, optional = _describe(annotation)
        has_default = field.default is not dataclasses.MISSING
        has_factory = field.default_factory is not dataclasses.MISSING
        params.append(
            ParamSpec(
                name=field.name,
                type=control,
                required=not (has_default or has_factory or optional),
                default=field.default if has_default else None,
                choices=choices,
                optional=optional,
                annotation="" if control != "unsupported" else str(annotation),
            )
        )
    return ComponentSpec(
        kind=kind, implementation=implementation, summary=_summary(cls), params=tuple(params)
    )


class ComponentCatalog(FrozenModel):
    """Everything a launcher form needs to know about what can be composed."""

    features: tuple[ComponentSpec, ...] = Field(default_factory=tuple)
    transforms: tuple[ComponentSpec, ...] = Field(default_factory=tuple)
    strategies: tuple[ComponentSpec, ...] = Field(default_factory=tuple)


def component_catalog() -> ComponentCatalog:
    """Describe every registered component."""
    return ComponentCatalog(
        features=tuple(describe(k, c) for k, c in sorted(FEATURES.items())),
        transforms=tuple(describe(k, c) for k, c in sorted(TRANSFORMS.items())),
        strategies=tuple(describe(k, c) for k, c in sorted(STRATEGIES.items())),
    )
