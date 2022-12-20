import numpy as np
import pygmo
import xarray as xr

from .constants import DESIGN_ID


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

    def _da(name, var, value):
        assert var.dims[0] == DESIGN_ID
        val = np.broadcast_to(value, var.shape[1:])

        return xr.DataArray(name=name, data=val, dims=var.dims[1:])

    scaler_ds = xr.Dataset(
        {
            name: _da(
                name=name,
                var=var,
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
                var=var,
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
    constraints = constraint_space(ds)
    eq_constraints = constraints.filter_by_attrs(
        type=lambda x: x and x["constraint"]["equals"] is not None
    )
    ineq_constraints = constraints.filter_by_attrs(
        type=lambda x: x and x["constraint"]["equals"] is None
    )

    if eq_constraints:
        raise NotImplementedError("Equality constraints are not supported yet")

    if ineq_constraints:
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

        ineq_feasibility_per_constraint = np.logical_and(
            ineq_constraints >= lower_bound_ds, ineq_constraints <= upper_bound_ds
        ).to_array()

        # Applies all() on all dimensions except DESIGN_ID
        ineq_feasibility_per_design = ineq_feasibility_per_constraint.groupby(
            DESIGN_ID
        ).all(...)

        ds = ds.where(ineq_feasibility_per_design, drop=True)

    return ds


def pareto_subset(ds):
    # objectives = ds.filter_by_attrs(role=VariableRole.OBJECTIVE)
    if len(ds[DESIGN_ID]) < 1:
        return ds

    scaled_objectives = (
        objective_space(ds, scale=True)
        .unstack()
        .to_stacked_array("weights", sample_dims=[DESIGN_ID])
    )

    pareto_mask = xr.DataArray(
        is_pareto_efficient(scaled_objectives.values), dims=[DESIGN_ID]
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

    ineq_cv_per_constraint = np.fabs(
        np.fmax(upper_bound_da, ineq_constraints_ds) / upper_bound_da_eps - 1
    ) + np.fabs(np.fmin(lower_bound_da, ineq_constraints_ds) / lower_bound_da_eps - 1)

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
