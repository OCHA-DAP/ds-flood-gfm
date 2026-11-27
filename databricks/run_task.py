"""Databricks entry wrapper for the repo's scripts.

Runs one of the pure-Python ``scripts/*.py`` files unchanged inside a
Databricks job task. The scripts and ``src/`` don't know about Databricks —
the same files run from a laptop — so this wrapper is the only
Databricks-specific glue:

1. Sets ``STAGE`` (the ocha-stratus data plane: DEV vs PROD DB + blob) plus
   any ``--env`` extras. Credentials are NOT set here: the Job Compute policy
   injects ``DSCI_AZ_*`` from the ``dsci`` secret scope; any extra secret the
   job needs is read from that scope at run time with ``--secret NAME`` (a
   missing key fails the task with a readable message rather than stopping
   the cluster launch, which is what a ``spark_env_vars`` reference does).
2. Copies ``src`` + ``scripts`` from the wsfs git checkout onto local disk and
   runs from there: importing packages straight off the workspace FUSE mount
   is unreliable (import probing intermittently raises filesystem errors).
3. Shells out to the script with ``PYTHONPATH`` at the copied ``src/`` so
   ``import ds_flood_gfm`` resolves without ``pip install -e .``. ``configs/``
   and ``results/`` (the fusion model tables) are copied alongside.

Adapted from ds-precip-intercomparison/databricks/run_task.py (team pattern,
KB infrastructure/databricks.md "The generic wrapper").

Usage (as the ``spark_python_task`` parameters):

    run_task.py scripts/fusion/daily_job.py --stage dev -- --cold-start
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

_COPY_DIRS = ("src", "scripts", "configs", "results")
_LOCAL_DIR = "ds_flood_gfm_run"


def _find_script_dir() -> str:
    """spark_python_task's exec context doesn't always define __file__."""
    try:
        return os.path.dirname(os.path.abspath(__file__))  # noqa: F821
    except NameError:
        pass
    if sys.argv and sys.argv[0]:
        return os.path.dirname(os.path.abspath(sys.argv[0]))
    return os.getcwd()


def _parse(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("script", help="script to run, relative to repo root")
    ap.add_argument(
        "--stage",
        required=True,
        choices=["dev", "prod"],
        help="ocha-stratus data plane (DEV vs PROD DB + blob)",
    )
    ap.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="extra env var for the script (repeatable)",
    )
    ap.add_argument(
        "--secret",
        action="append",
        default=[],
        metavar="NAME",
        help="dsci-scope secret to expose as env var NAME (repeatable); missing = fail",
    )
    ap.add_argument(
        "--optional-secret",
        action="append",
        default=[],
        metavar="NAME",
        help="like --secret, but a missing key only logs a warning",
    )
    ap.epilog = "Everything after a literal `--` is passed through to the script."
    # Split on the first literal "--" ourselves: an argparse REMAINDER
    # positional would swallow --stage/--env as soon as it sees the script.
    if "--" in argv:
        i = argv.index("--")
        argv, script_args = argv[:i], argv[i + 1 :]
    else:
        script_args = []
    args = ap.parse_args(argv)
    args.script_args = script_args
    return args


def _resolve_secrets(env, names, optional=False):
    """Pull extra secrets from the dsci scope at run time (not via
    spark_env_vars): a missing or misnamed key then fails THIS task with a
    readable message instead of stopping the job cluster from launching."""
    if not names:
        return
    from pyspark.dbutils import DBUtils
    from pyspark.sql import SparkSession

    dbutils = DBUtils(SparkSession.builder.getOrCreate())
    for name in names:
        try:
            env[name] = dbutils.secrets.get(scope="dsci", key=name)
        except Exception as exc:  # noqa: BLE001 — the SDK raises a generic Py4J error
            if optional:
                print(f"[run_task] optional secret dsci/{name} not found; continuing without it")
                continue
            raise RuntimeError(f"secret dsci/{name} is missing from the dsci scope: {exc}") from exc


def main(argv=None):
    args = _parse(sys.argv[1:] if argv is None else argv)
    repo_root = os.path.abspath(os.path.join(_find_script_dir(), ".."))
    # Per-process copy: tasks sharing a job cluster start concurrently, and one
    # task's copytree must not race another task's imports.
    local_root = os.path.join(
        "/local_disk0" if os.path.isdir("/local_disk0") else tempfile.gettempdir(),
        f"{_LOCAL_DIR}_{os.getpid()}",
    )
    for sub in _COPY_DIRS:
        shutil.copytree(
            os.path.join(repo_root, sub),
            os.path.join(local_root, sub),
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )

    env = dict(os.environ)
    env["STAGE"] = args.stage
    for kv in args.env:
        key, _, value = kv.partition("=")
        if not key:
            raise ValueError(f"bad --env {kv!r}; expected KEY=VALUE")
        env[key] = value
    _resolve_secrets(env, args.secret)
    _resolve_secrets(env, args.optional_secret, optional=True)
    env["PYTHONPATH"] = os.path.join(local_root, "src") + os.pathsep + env.get("PYTHONPATH", "")
    # Unbuffered so the script's prints interleave correctly in the run log.
    env["PYTHONUNBUFFERED"] = "1"

    cmd = [sys.executable, os.path.join(local_root, args.script), *args.script_args]
    shown = {k: env[k] for k in ["STAGE", *[kv.partition("=")[0] for kv in args.env]]}
    print(f"[run_task] script={args.script} env={shown} "
          f"secrets={args.secret + args.optional_secret} args={args.script_args}")
    rc = subprocess.run(cmd, cwd=local_root, env=env, check=False).returncode
    # Databricks treats a top-level SystemExit (even code 0) as a task failure;
    # raise only on non-zero and let success return naturally.
    if rc != 0:
        raise RuntimeError(f"{args.script} exited with code {rc}")
    print("[run_task] OK")


if __name__ == "__main__":
    main()
