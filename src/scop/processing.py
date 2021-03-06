import numpy as np
import pandas as pd
import pygmo
import xarray as xr

from .constants import DESIGN_ID


def xr_value_dims(name, value):
    """
    Returns the value and dimensions/coordinates in the way we like 'em.

    >>> import xarray as xr
    >>> name = "foo"
    >>> value = np.array([1, 2, 3])
    >>> xr_value, xr_dims = xr_value_dims(name, value)
    >>> xr.DataArray(
    ...     name=name,
    ...     data=xr_value,
    ...     coords=xr_dims,
    ... )
    <xarray.DataArray 'foo' (foo_dim: 3)>
    array([1, 2, 3])
    Coordinates:
      * foo_dim          (foo_dim) MultiIndex
      - foo_dim_level_0  (foo_dim) int64 0 1 2
    """
    val = np.atleast_1d(value).copy()
    extra_dims = []
    if val.size > 1:
        idx = pd.MultiIndex.from_tuples(np.ndindex(val.shape))
        extra_dims = [(f"{name}_dim", idx)]
    val = val.reshape((-1,))
    return val, extra_dims


def _is_pareto_efficient(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A boolean array of pareto-efficient points.
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    # Next index in the is_efficient array to search for
    next_point_index = 0
    while next_point_index < len(costs):
        nondominated_point_mask = np.any(costs < costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        # Removes dominated points
        is_efficient = is_efficient[nondominated_point_mask]
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1

    is_efficient_mask = np.zeros(n_points, dtype=bool)
    is_efficient_mask[is_efficient] = True
    return is_efficient_mask


def is_pareto_efficient(costs):
    ixs = np.argsort(
        ((costs - costs.mean(axis=0)) / (costs.std(axis=0) + 1e-7)).sum(axis=1)
    )
    costs = costs[ixs]
    is_efficient = _is_pareto_efficient(costs)
    is_efficient[ixs] = is_efficient.copy()
    return is_efficient


def design_space(ds):
    return ds.filter_by_attrs(type=lambda x: x and "desvar" in x)


def objective_space(ds, scale=False):
    objectives = ds.filter_by_attrs(type=lambda x: x and "objective" in x)
    if not scale:
        return objectives

    def _da(name, value):
        value, dims = xr_value_dims(name, value)
        return xr.DataArray(
            name=name, data=value.item() if not dims else value, coords=dims or None
        )

    scaler_ds = xr.Dataset(
        {
            name: _da(
                name=name,
                value=(
                    # We cannot use a simple (x or 1.0) since x might be an array
                    var.attrs["type"]["objective"]["scaler"]
                    if var.attrs["type"]["objective"]["scaler"] is not None
                    else 1.0
                ),
            )
            for (name, var) in objectives.items()
        }
    )
    adder_ds = xr.Dataset(
        {
            name: _da(
                name=name,
                value=(
                    # We cannot use a simple (x or 0.0) since x might be an array
                    var.attrs["type"]["objective"]["adder"]
                    if var.attrs["type"]["objective"]["adder"] is not None
                    else 0.0
                ),
            )
            for (name, var) in objectives.items()
        }
    )

    return objectives * scaler_ds + adder_ds


def constraint_space(ds):
    return ds.filter_by_attrs(type=lambda x: x and "constraint" in x)


def feasible_subset(ds):
    # objectives = ds.filter_by_attrs(role=VariableRole.OBJECTIVE)
    constraints = constraint_space(ds)
    eq_constraints = constraints.filter_by_attrs(
        type=lambda x: x and x["constraint"]["equals"] is not None
    )
    ineq_constraints = constraints.filter_by_attrs(
        type=lambda x: x and x["constraint"]["equals"] is None
    )

    lower_bound_ds = xr.Dataset(
        {
            name: var.attrs["type"]["constraint"]["lower"]
            for (name, var) in ineq_constraints.items()
        }
    )
    upper_bound_ds = xr.Dataset(
        {
            name: var.attrs["type"]["constraint"]["upper"]
            for (name, var) in ineq_constraints.items()
        }
    )

    ineq_feasibility_per_constraint = xr.ufuncs.logical_and(
        ineq_constraints >= lower_bound_ds, ineq_constraints <= upper_bound_ds
    ).to_array()

    # Applies all() on all dimensions except DESIGN_ID
    ineq_feasibility_per_design = ineq_feasibility_per_constraint.groupby(
        DESIGN_ID
    ).all(...)

    return ds.where(ineq_feasibility_per_design, drop=True)


def pareto_subset(ds):
    # objectives = ds.filter_by_attrs(role=VariableRole.OBJECTIVE)
    if len(ds[DESIGN_ID]) < 1:
        raise ValueError("Supplied dataset has no designs.")
    scaled_objectives = objective_space(ds, scale=True).to_array()
    pareto_mask = xr.DataArray(
        is_pareto_efficient(scaled_objectives.T.values), dims=[DESIGN_ID]
    )

    return ds.where(pareto_mask, drop=True)


def epsilonify(da: xr.DataArray, eps=np.finfo(float).eps) -> xr.DataArray:
    da = da.copy()
    da[da.isin([0.0])] = eps
    return da


def constraint_violations(ds):
    # FIXME: support equality constraints
    # eq_constraints_ds = ds.filter_by_attrs(
    #     type=lambda x: x and "constraint" in x and x["constraint"]["equals"] is not None
    # )
    ineq_constraints_ds = ds.filter_by_attrs(
        type=lambda x: x and "constraint" in x and x["constraint"]["equals"] is None
    )

    lower_bound_da = xr.Dataset(
        {
            name: var.attrs["type"]["constraint"]["lower"]
            for (name, var) in ineq_constraints_ds.items()
        }
    ).to_array()
    lower_bound_da_eps = epsilonify(lower_bound_da)
    upper_bound_da = xr.Dataset(
        {
            name: var.attrs["type"]["constraint"]["upper"]
            for (name, var) in ineq_constraints_ds.items()
        }
    ).to_array()
    upper_bound_da_eps = epsilonify(upper_bound_da)

    ineq_cv_per_constraint = xr.ufuncs.fabs(
        xr.ufuncs.fmax(upper_bound_da, ineq_constraints_ds) / upper_bound_da_eps - 1
    ) + xr.ufuncs.fabs(
        xr.ufuncs.fmin(lower_bound_da, ineq_constraints_ds) / lower_bound_da_eps - 1
    )

    # Applies sum() on all dimensions except DESIGN_ID
    ineq_feasibility_per_design = (
        ineq_cv_per_constraint.to_array().groupby(DESIGN_ID).sum(...)
    )

    # Arranges the array in the same order as the input
    return ineq_feasibility_per_design.sel({DESIGN_ID: ds[DESIGN_ID]})


def annotate_ds_with_constraint_violations(ds):
    cv = constraint_violations(ds)
    return ds.merge({"constraint_violation": cv})


def hv_ref_point(ds, offset_ratio=0.001):
    scaled_objectives = objective_space(ds, scale=True)

    nadir_point = scaled_objectives.max()
    ref_point = nadir_point + abs(nadir_point) * offset_ratio

    return ref_point.to_array()


def hypervolume(ds, ref_point=None):
    scaled_objectives = objective_space(ds, scale=True)

    hv = pygmo.hypervolume(scaled_objectives.to_array().T)

    if ref_point is None:
        ref_point = hv.refpoint()

    return xr.DataArray(
        hv.compute(ref_point), name="hypervolume", attrs={"units": None}
    )
