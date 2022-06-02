import itertools
import warnings
from collections import OrderedDict
from itertools import chain

import numpy as np
import pandas as pd
import xarray as xr
from openmdao.core.driver import Driver
from openmdao.core.problem import Problem
from openmdao.core.system import System
from openmdao.recorders.case_recorder import CaseRecorder
from openmdao.solvers.solver import Solver

from .constants import DESIGN_ID

SEMVAR_PREFIX = "semvar:"


def generate_abs2meta(recording_requester, semvar_registry=None):
    meta = {}
    ##### START ADAPTATION FROM SqliteRecorder #####
    driver = None

    # grab the system
    if isinstance(recording_requester, Driver):
        system = recording_requester._problem().model
        driver = recording_requester
    elif isinstance(recording_requester, System):
        system = recording_requester
    elif isinstance(recording_requester, Problem):
        system = recording_requester.model
        driver = recording_requester.driver
    elif isinstance(recording_requester, Solver):
        system = recording_requester._system()
    else:
        raise ValueError(
            "Driver encountered a recording_requester it cannot handle"
            ": {0}".format(recording_requester)
        )

    states = system._list_states_allprocs()

    if driver is None:
        desvars = system.get_design_vars(True, get_sizes=False)
        responses = system.get_responses(True, get_sizes=False)
        objectives = OrderedDict()
        constraints = OrderedDict()
        for name, data in responses.items():
            if data["type"] == "con":
                constraints[name] = data
            else:
                objectives[name] = data
    else:
        desvars = driver._designvars
        constraints = driver._cons
        objectives = driver._objs
        responses = driver._responses

    inputs = list(system.abs_name_iter("input", local=False, discrete=True))
    outputs = list(system.abs_name_iter("output", local=False, discrete=True))

    full_var_set = [
        (desvars, "desvar"),
        (responses, "response"),
        (objectives, "objective"),
        (constraints, "constraint"),
    ]

    # # merge current abs2prom and prom2abs with this system's version
    # self._abs2prom["input"].update(system._var_abs2prom["input"])
    # self._abs2prom["output"].update(system._var_abs2prom["output"])
    # for v, abs_names in system._var_allprocs_prom2abs_list["input"].items():
    #     if v not in self._prom2abs["input"]:
    #         self._prom2abs["input"][v] = abs_names
    #     else:
    #         self._prom2abs["input"][v] = list(
    #             set(chain(self._prom2abs["input"][v], abs_names))
    #         )

    # # for outputs, there can be only one abs name per promoted name
    # for v, abs_names in system._var_allprocs_prom2abs_list["output"].items():
    #     self._prom2abs["output"][v] = abs_names

    # absolute pathname to metadata mappings for continuous & discrete variables
    # discrete mapping is sub-keyed on 'output' & 'input'
    real_meta = system._var_allprocs_abs2meta
    disc_meta = system._var_allprocs_discrete

    for kind, discrete in itertools.product(["input", "output"], [True, False]):
        vars_meta = disc_meta if discrete else real_meta
        for name, data in vars_meta[kind].items():
            meta[name] = data.copy()
            meta[name]["discrete"] = discrete
            meta[name]["type"] = {kind: {}}
            meta[name]["explicit"] = kind == "input" or name not in states

    for var_set, var_type in full_var_set:
        for name in var_set:
            # Design variables can be requested by input name.
            if var_type == "desvar":
                name = var_set[name]["ivc_source"]

            if var_type not in meta[name]["type"]:
                try:
                    var_type_meta = var_set[name]
                except (KeyError, TypeError):
                    var_type_meta = {}
                meta[name]["type"][var_type] = var_type_meta

    if semvar_registry:
        for name, data in meta.items():
            tags = data.get("tags", set())
            for tag in tags:
                _, match, semvar_name = tag.partition(SEMVAR_PREFIX)
                if match:
                    data["semvar"] = semvar_registry.variables[semvar_name]
                    break
            else:
                data["semvar"] = None

    return meta


def get_dim_name(meta, name, idx):
    semvar = meta.get("semvar", None)
    if semvar:
        if semvar.dims is not None:
            try:
                return semvar.dims[idx]
            except IndexError:
                warnings.warn(
                    f"Semvar '{semvar.name}' defines {len(semvar.dims)} dimension names, but variable '{name}' is using more dimensions. Falling back to default dimension name."
                )
        return f"{semvar.name}_{idx}"
    return f"{name}_{idx}"


def make_data_vars(all_vars, all_meta, design_idx):
    for (name, value) in all_vars.items():
        meta = all_meta[name]
        val = np.atleast_1d(value).copy()
        if val.size > 1:
            # Why coords with simple integer indexes? Answer: it makes
            # it possible to merge all datasets even though the
            # dimensions have different sizes.
            extra_coords = {
                get_dim_name(meta, name, idx): range(size)
                for idx, size in enumerate(val.shape)
            }
            val = val[np.newaxis, ...]
        else:
            extra_coords = {}

        coords = {DESIGN_ID: design_idx, **extra_coords}
        dims = list(coords.keys())

        yield ((name, (dims, val, meta)), coords.items())


class DatasetRecorder(CaseRecorder):
    def __init__(self, record_viewer_data=False, semvar_registry=None):
        if record_viewer_data:
            raise NotImplementedError(
                "This recorder does not support recording of metadata for viewing."
            )
        super().__init__(record_viewer_data=record_viewer_data)
        self.datasets = {}
        # self._abs2prom = {"input": {}, "output": {}}
        # self._prom2abs = {"input": {}, "output": {}}
        self._abs2meta = {}
        self.semvar_registry = semvar_registry

    def startup(self, recording_requester):
        super().startup(recording_requester)
        # ds = xr.Dataset(data_vars={"counter": xr.DataArray(), "timestamp": xr.DataArray()}, coords={"name": xr.DataArray()})
        self.datasets[recording_requester] = []

        self._abs2meta.update(
            generate_abs2meta(recording_requester, semvar_registry=self.semvar_registry)
        )

    def record_iteration_driver(self, recording_requester, data, metadata):
        timestamp = pd.Timestamp.fromtimestamp(metadata["timestamp"])

        all_vars = dict(sorted(chain(data["input"].items(), data["output"].items())))

        # hvplot borks of MultiIndex :((
        # design_idx = pd.MultiIndex.from_tuples(
        #     [(metadata["name"], 0, self._counter - 1, self._iteration_coordinate)],
        #     names=("driver", "rank", "counter", "name"),
        # )
        design_idx = np.array([self._iteration_coordinate])

        meta_vars = {
            key: xr.DataArray([item], dims=[DESIGN_ID])
            for (key, item) in metadata.items()
            if key not in ["name", "success", "timestamp", "msg"]
        }
        data_vars_pairs, coords_pairs = zip(
            *make_data_vars(all_vars, self._abs2meta, design_idx)
        )
        data_vars = dict(data_vars_pairs)
        coords = dict(itertools.chain(*coords_pairs))

        ds = xr.Dataset(
            data_vars={
                "timestamp": xr.DataArray([timestamp], dims=[DESIGN_ID]),
                "success": xr.DataArray([bool(metadata["success"])], dims=[DESIGN_ID]),
                "msg": xr.DataArray([metadata["msg"]], dims=[DESIGN_ID]),
                **meta_vars,
                **data_vars,
            },
            coords=coords,
        )

        self.datasets[recording_requester].append(ds)

    def record_iteration_problem(self, recording_requester, data, metadata):
        raise NotImplementedError(
            "This recorder does not support recording of problems."
        )

    def record_iteration_solver(self, recording_requester, data, metadata):
        raise NotImplementedError(
            "This recorder does not support recording of solvers."
        )

    def record_iteration_system(self, recording_requester, data, metadata):
        raise NotImplementedError(
            "This recorder does not support recording of systems."
        )

    def record_derivatives_driver(self, recording_requester, data, metadata):
        raise NotImplementedError(
            "This recorder does not support recording of derivatives."
        )

    def record_metadata_solver(self, solver, run_number=None):
        pass

    def record_metadata_system(self, system, run_number=None):
        pass

    def record_viewer_data(self, model_viewer_data):
        pass

    def assemble_dataset(self, recording_requester):
        return xr.concat(self.datasets[recording_requester], dim=DESIGN_ID)
