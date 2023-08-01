from invoke import task

DEFAULT_ENV_NAME = "openmdao-scop"


@task
def conda_run(c, cmd, env_name=DEFAULT_ENV_NAME):
    c.run(f"conda run -n {env_name} {cmd}")


@task
def setup(c, env_name=DEFAULT_ENV_NAME):
    """
    Installs the Python environment.
    """
    c.run(f"conda lock install -n {env_name} conda-lock.yml")
    conda_run(c, f"pip install --no-deps -e .", env_name=env_name)


@task
def lock(c):
    c.run("conda lock -f conda-recipe/meta.yaml --lockfile conda-lock.yml")
