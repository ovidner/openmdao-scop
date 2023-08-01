from typing import Any, Optional

import numpy as np
import openmdao.api as om
from openmdao.core.component import Component
from pydantic import BaseModel, Field

NOT_SET = object()


class Space(BaseModel):
    """
    Base class for representing a space of values. Don't use this directly.
    """

    pass


class InnumSpace(Space):
    """
    A space of innumerable values, such as arbitrary strings or objects.
    """

    pass


class EnumSpace(Space):
    """
    A space of enumerable values, such as discrete numbers, strings or booleans.
    """

    values: list[Any] = Field(default_factory=list)
    ordered: bool = Field(default=False)


class BoundedSpace(Space):
    lower: Optional[Any] = Field(default=None)
    upper: Optional[Any] = Field(default=None)

    # TODO: add validation that lower < upper
    # TODO: maybe reconsider this inheritance structure, or maybe use generics

    def __init__(self, lower=None, upper=None, **kwargs):
        super().__init__(lower=lower, upper=upper, **kwargs)


class RealSpace(BoundedSpace):
    """
    A space of real numbers, with optional lower and upper bounds.
    """

    lower: Optional[float] = None
    upper: Optional[float] = None


class IntegerSpace(BoundedSpace):
    """
    A space of integers, with optional lower and upper bounds.
    """

    lower: Optional[int] = None
    upper: Optional[int] = None


def bool_space():
    return EnumSpace(values=[False, True])


class Param(BaseModel):
    name: str = Field(description="Unique name.")
    label: Optional[str] = Field(description="Short, human-friendly name.")
    desc: Optional[str] = Field(description="Longer description.")

    default: Optional[Any] = Field(default=None, description="Default value.")
    # I would rather call this "unit", but let's stick to OpenMDAO's convention
    units: Optional[str] = Field(default=None)
    space: Optional[Space] = Field(default_factory=RealSpace)

    discrete: bool = Field(
        default=False,
        description="Does this parameter have discrete properties, as OpenMDAO understands them?",
    )

    tags: list[str] = Field(
        default_factory=list, description="Set of tags, for OpenMDAO compatibility."
    )
    meta: dict = Field(
        default_factory=dict,
        description="Metadata dictionary. Use for arbitrary good, or evil, purposes.",
    )
    parent: Optional["Param"] = Field(
        default=None, description="Parent parameter for overridden params."
    )

    class Config:
        allow_mutation = False

    def override(self, **kwargs):
        # TODO: warn if overriding the unit on a discrete param, as no
        # unit conversion will be automatically performed

        # Copy the meta dict if not supplied, so that we don't modify
        # the original
        kwargs["meta"] = kwargs.pop("meta", self.meta.copy())

        args = {**self.dict(), **kwargs, "parent": self}
        return Param(**args)


class ParamSet(dict):
    def __init__(self, params: list[Param]):
        super().__init__((p.name, p) for p in params)

    def __repr__(self) -> str:
        # Somehow, super().__repr__() doesn't work here
        return f"ParamSet({dict.__repr__(self)})"

    def add(self, param: Param):
        self[param.name] = param


def get_scop_meta(comp: Component):
    if not hasattr(comp, "_scop_meta"):
        comp._scop_meta = {"inputs": {}, "outputs": {}}
    return comp._scop_meta


def add_input_param(comp: Component, param: Param):
    if param.discrete:
        comp.add_discrete_input(
            name=param.name, val=param.default, desc=param.desc, tags=param.tags
        )
    else:
        comp.add_input(
            name=param.name,
            val=param.default,
            units=param.units,
            desc=param.desc,
            tags=param.tags,
        )

    meta = get_scop_meta(comp)
    meta["inputs"][param.name] = param


def add_output_param(comp: Component, param: Param):
    if param.discrete:
        comp.add_discrete_output(
            name=param.name, val=param.default, desc=param.desc, tags=param.tags
        )
    else:
        comp.add_output(
            name=param.name,
            val=param.default,
            units=param.units,
            desc=param.desc,
            tags=param.tags,
        )

    meta = get_scop_meta(comp)
    meta["outputs"][param.name] = param
