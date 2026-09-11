"""Background jobs: the UI submits work, a subprocess does it.

See ``docs/workbench_plan.md``. The runner spawns the CLI rather than calling the engine
in-process, which is what makes the reproducibility rule in ``docs/decisions.md`` D46
structural: the UI's only power is to write a config and run the command you would run.
"""

from qresearch.jobs.contracts import JobKind, JobRecord, JobState
from qresearch.jobs.runner import JobRunner
from qresearch.jobs.store import JobStore

__all__ = ["JobKind", "JobRecord", "JobRunner", "JobState", "JobStore"]
