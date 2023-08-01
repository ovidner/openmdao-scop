import numpy as np
import openmdao.api as om
import pandas as pd
import scop


def test_recording_timestamps():
    prob = om.Problem()
    model = prob.model

    indeps = model.add_subsystem("indeps", om.IndepVarComp(), promotes=["*"])
    indeps.add_output("cont", 1.0)
    indeps.add_discrete_output("int", 1)

    model.add_design_var("cont", lower=-1.0, upper=1)
    model.add_design_var("int", lower=-1, upper=1)

    driver = om.DOEDriver(
        om.ListGenerator(
            [
                [("cont", -1.0), ("int", -1)],
                [("cont", 0.0), ("int", 0)],
                [("cont", 1.0), ("int", 1)],
            ]
        )
    )
    recorder = scop.DatasetRecorder()
    driver.add_recorder(recorder)
    driver.recording_options["includes"] = ["*"]

    prob.driver = driver

    pre_timestamp = pd.Timestamp.utcnow().to_numpy()
    prob.setup()
    prob.run_driver()
    post_timestamp = pd.Timestamp.utcnow().to_numpy()

    ds = recorder.assemble_dataset(driver)

    assert ds.attrs["start_timestamp"] >= pre_timestamp
    assert ds.attrs["start_timestamp"] <= post_timestamp
    assert ds["meta.timestamp"].dtype == "datetime64[ns]"
    assert ds["meta.timestamp"].min() >= pre_timestamp
    assert ds["meta.timestamp"].max() <= post_timestamp
    # Make sure timestamps are monotonically increasing
    assert (ds["meta.timestamp"].diff("design") > np.timedelta64(0, "ns")).all()
