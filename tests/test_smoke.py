import itertools

import deepdiff
import numpy as np
import openmdao.api as om
import pytest
from xarray.testing import assert_equal

import scop
from scop import DESIGN_ID, DatasetRecorder, pareto_subset


def nans(shape):
    return np.ones(shape) * np.nan


def assert_ds_equal(a, b):
    def _all_attrs(x):
        return {name: var.attrs for name, var in x.variables.items()}

    assert_equal(a, b)
    assert not deepdiff.DeepDiff(a.attrs, b.attrs)
    assert not deepdiff.DeepDiff(
        _all_attrs(a),
        _all_attrs(b),
    )
    assert a.variables.keys() == b.variables.keys()


@pytest.mark.parametrize("weights", itertools.product((1.0, -1.0), repeat=3))
def test_pareto_dataset(weights):
    weights_arr = np.array(weights)
    var_shape = (3,)
    prob = om.Problem()
    prob.model.add_subsystem("indeps", om.IndepVarComp("x", nans(var_shape)))
    prob.model.add_subsystem(
        "passthrough",
        om.ExecComp(
            ["y1=x[0]", "y2=x[1:3]"], x=nans((3,)), y1=nans((1,)), y2=nans((2,))
        ),
    )
    prob.model.connect(
        "indeps.x",
        "passthrough.x",
    )

    prob.model.add_design_var(
        "indeps.x", lower=np.zeros(var_shape), upper=np.ones(var_shape)
    )
    prob.model.add_objective("passthrough.y1", scaler=weights_arr[0])
    prob.model.add_objective("passthrough.y2", scaler=weights_arr[1:3])

    prob.driver = driver = om.DOEDriver(
        om.ListGenerator(
            [
                [("indeps.x", np.array(x))]
                for x in [
                    [0, 0, 0],
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [-1, 0, 0],
                    [0, -1, 0],
                    [0, 0, -1],
                ]
            ]
        )
    )

    recorder = DatasetRecorder()
    driver.add_recorder(recorder)

    try:
        prob.setup()
        prob.run_driver()
    finally:
        prob.cleanup()

    ds = recorder.assemble_dataset(driver)
    assert len(ds[DESIGN_ID]) == 7

    pareto_ds = pareto_subset(ds)
    assert pareto_ds is not ds
    assert len(pareto_ds[DESIGN_ID]) == 3
    expected_pareto_set = -np.eye(3) * weights_arr

    # It is more convenient to simply look at the input vectors than the output
    # dito. We assume everything works in between.
    assert np.all(np.isin(expected_pareto_set, pareto_ds["indeps.x"]))


def test_dump_load(tmp_path):
    var_shape = (3,)
    prob = om.Problem()
    indeps = prob.model.add_subsystem("indeps", om.IndepVarComp("x", nans(var_shape)))
    # Add a few troublesome outputs
    indeps.add_discrete_output("bool", True)
    indeps.add_discrete_output("array", np.array([1, 2, 3]))
    indeps.add_discrete_output("str", "foo")
    indeps.add_discrete_output("name:unsafe", 0)
    prob.model.add_subsystem(
        "passthrough",
        om.ExecComp(
            ["y1=x[0]", "y2=x[1:3]"], x=nans((3,)), y1=nans((1,)), y2=nans((2,))
        ),
    )
    prob.model.connect(
        "indeps.x",
        "passthrough.x",
    )

    prob.model.add_design_var(
        "indeps.x", lower=np.zeros(var_shape), upper=np.ones(var_shape)
    )

    prob.driver = driver = om.DOEDriver(
        om.ListGenerator(
            [
                [("indeps.x", np.array(x))]
                for x in [
                    [0, 0, 0],
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [-1, 0, 0],
                    [0, -1, 0],
                    [0, 0, -1],
                ]
            ]
        )
    )

    recorder = scop.DatasetRecorder()
    driver.recording_options["includes"] = ["*"]
    driver.add_recorder(recorder)

    try:
        prob.setup()
        prob.run_driver()
    finally:
        prob.cleanup()

    ds = recorder.assemble_dataset(driver)
    ds_copy = ds.copy(deep=True)

    path = tmp_path / "dump.scop"

    scop.dump(ds, path)

    # Make sure we don't mutate the ds by dumping it
    assert_ds_equal(ds, ds_copy)

    dumped_and_loaded_ds = scop.load(path)

    assert_ds_equal(dumped_and_loaded_ds, ds)
