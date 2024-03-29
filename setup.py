from setuptools import find_packages, setup

setup(
    name="openmdao-scop",
    use_scm_version=True,
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    python_requires=">=3.7, <4",
    setup_requires=["setuptools_scm"],
    # FIXME: this should also list pygmo as a requirement. However, pygmo lacks
    # the correct metadata even when installed with Conda, so it will just break.
    install_requires=[
        "jsonpickle",
        "openmdao",
        "numpy",
        "pandas",
        "pydantic",
        "xarray",
        "zarr",
    ],
)
