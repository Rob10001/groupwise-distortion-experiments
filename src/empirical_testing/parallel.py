"""Bounded process parallelism with results returned in task order.

Callers assign task identities and random seeds before dispatch. Workers return
their results to the parent, which aggregates them in the original order.
Failures name the task and preserve the original exception; serial fallback is
opt-in.

Workers inherit the parent's thread settings by default. Changing BLAS thread
counts can affect the numerical optimizer, so thread limiting is explicit.
Windows uses spawn: worker functions must be module-level callables, and large
shared inputs can be loaded once per worker through an initializer.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

__all__ = [
    "BLAS_THREAD_VARIABLES",
    "MapReport",
    "WorkerFailure",
    "blas_thread_budget",
    "ordered_map",
    "resolve_workers",
]

#: Environment variables every common CPU BLAS/OpenMP backend reads at import.
BLAS_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


class WorkerFailure(RuntimeError):
    """A task raised inside a worker process.

    Carries the task's position and its repr so a failure points at the cell
    that produced it, and chains the worker's exception as ``__cause__`` so the
    original traceback survives.
    """

    def __init__(self, index: int, task: Any, cause: BaseException):
        self.index = int(index)
        self.task = task
        super().__init__(
            f"task {index} ({task!r}) failed in a worker process: "
            f"{type(cause).__name__}: {cause}"
        )


@dataclass
class MapReport:
    """What :func:`ordered_map` actually did, for logging and provenance.

    Kept separate from the results so a caller can record how the work ran
    without that leaking into what the work produced.  ``worker_seconds`` is the
    summed wall time the tasks themselves took; it is *not* a CPU-time
    measurement, and the parent cannot observe worker CPU time, so nothing here
    should be reported as such.
    """

    requested_workers: int
    used_workers: int
    n_tasks: int
    wall_seconds: float = 0.0
    startup_seconds: float = 0.0
    worker_seconds: float = 0.0
    aggregation_seconds: float = 0.0
    fell_back_to_serial: bool = False
    fallback_reason: str = ""
    per_task_seconds: list[float] = field(default_factory=list)

    def summary(self) -> str:
        mode = "serial" if self.used_workers == 1 else f"{self.used_workers} workers"
        note = ""
        if self.fell_back_to_serial:
            note = f" [fell back to serial: {self.fallback_reason}]"
        return (
            f"{self.n_tasks} task(s), {mode}, wall {self.wall_seconds:.1f}s "
            f"(pool startup {self.startup_seconds:.1f}s, task time "
            f"{self.worker_seconds:.1f}s, aggregation "
            f"{self.aggregation_seconds:.1f}s){note}"
        )


def resolve_workers(requested: int, n_tasks: int, *, reserve: int = 0) -> int:
    """Clamp a worker request to the work available and to this machine.

    ``reserve`` leaves that many logical CPUs unclaimed, for a caller that is
    itself running inside an outer pool.  Note that a logical CPU count is not a
    promise of proportional throughput -- these workloads are memory- and
    interpreter-bound well before they are core-bound -- so this only prevents
    obvious oversubscription; the useful worker count is an empirical question,
    which is why the benchmarks sweep it.
    """

    requested = max(1, int(requested))
    if n_tasks <= 1:
        return 1
    available = os.cpu_count() or 1
    ceiling = max(1, available - max(0, int(reserve)))
    return max(1, min(requested, n_tasks, ceiling))


@contextmanager
def blas_thread_budget(threads_per_worker: int | None = 1):
    """Pin BLAS/OpenMP threads for processes *started inside this block*.

    ``None`` is a no-op: the environment is left exactly as it is, so children
    inherit the parent's threading and reproduce the parent's arithmetic.  See
    the module docstring for why that, not pinning, is the default.

    The variables are read by each backend when it is first imported.  The
    parent has already imported NumPy, so setting them here does not change the
    parent's own behaviour; it changes what freshly spawned children inherit,
    which is exactly the intent -- ``workers`` processes each using one BLAS
    thread rather than ``workers * n_cores`` threads fighting for the machine.
    The previous environment is restored on exit, including variables that were
    unset.
    """

    if threads_per_worker is None:
        yield
        return
    saved = {name: os.environ.get(name) for name in BLAS_THREAD_VARIABLES}
    value = str(max(1, int(threads_per_worker)))
    try:
        for name in BLAS_THREAD_VARIABLES:
            os.environ[name] = value
        yield
    finally:
        for name, previous in saved.items():
            if previous is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous


def _timed(function: Callable[[Any], Any], task: Any) -> tuple[Any, float]:
    mark = time.perf_counter()
    return function(task), time.perf_counter() - mark


def ordered_map(
    function: Callable[[Any], Any],
    tasks: Sequence[Any],
    workers: int = 1,
    *,
    threads_per_worker: int | None = None,
    allow_serial_fallback: bool = False,
    initializer: Callable[..., None] | None = None,
    initargs: tuple = (),
    on_progress: Callable[[int, int, Any, float], None] | None = None,
    log: Callable[[str], None] | None = None,
) -> tuple[list[Any], MapReport]:
    """Apply ``function`` to every task, returning results in **task order**.

    ``function`` must be a module-level callable taking one picklable argument
    (Windows spawns its children).  ``initializer``/``initargs`` run once per
    worker process and are the place to materialize a large read-only payload,
    so it is pickled once per worker instead of once per task; the serial path
    runs the same initializer, so both paths go through identical code.  Results are collected into a list indexed by
    task position, so the returned sequence -- and therefore anything the caller
    aggregates from it -- is identical for one worker, six workers, or a run
    that was interrupted and redone.

``threads_per_worker`` defaults to ``None``, meaning
    workers inherit the parent's BLAS/OpenMP threading; see the module
    docstring for why pinning is not the default.

    A worker exception propagates as :class:`WorkerFailure`.  If
    ``allow_serial_fallback`` is set, an error raised by the *pool itself*
    (process start-up, pickling, a killed child) is logged with its cause and
    the same tasks are then run serially in the same order; a
    :class:`WorkerFailure` -- a genuine error inside the task -- is re-raised
    either way, because a task that cannot compute its answer must not be
    retried into a slow success.
    """

    tasks = list(tasks)
    used = resolve_workers(workers, len(tasks))
    report = MapReport(requested_workers=max(1, int(workers)), used_workers=used,
                       n_tasks=len(tasks))
    if not tasks:
        return [], report

    emit = log or (lambda _message: None)
    started = time.perf_counter()

    def run_serially() -> list[Any]:
        results: list[Any] = []
        for position, task in enumerate(tasks):
            try:
                value, seconds = _timed(function, task)
            except Exception as error:  # the task itself failed
                raise WorkerFailure(position, task, error) from error
            results.append(value)
            report.per_task_seconds.append(seconds)
            report.worker_seconds += seconds
            if on_progress is not None:
                on_progress(position + 1, len(tasks), task, seconds)
        return results

    if used == 1:
        if initializer is not None:
            initializer(*initargs)
        results = run_serially()
        report.wall_seconds = time.perf_counter() - started
        return results, report

    results_by_index: list[Any] = [None] * len(tasks)
    seconds_by_index: list[float] = [0.0] * len(tasks)
    try:
        with blas_thread_budget(threads_per_worker):
            pool_mark = time.perf_counter()
            with ProcessPoolExecutor(
                max_workers=used, initializer=initializer, initargs=initargs
            ) as pool:
                futures = {
                    pool.submit(_timed, function, task): position
                    for position, task in enumerate(tasks)
                }
                report.startup_seconds = time.perf_counter() - pool_mark
                from concurrent.futures import as_completed

                for finished, future in enumerate(as_completed(futures), start=1):
                    position = futures[future]
                    try:
                        value, seconds = future.result()
                    except Exception as error:
                        # A task-level error. Never absorbed: a wrong answer
                        # must not be indistinguishable from a slow one.
                        raise WorkerFailure(position, tasks[position], error) from error
                    results_by_index[position] = value
                    seconds_by_index[position] = seconds
                    if on_progress is not None:
                        on_progress(finished, len(tasks), tasks[position], seconds)
    except WorkerFailure:
        raise
    except Exception as error:  # pool-level failure only
        if not allow_serial_fallback:
            raise
        report.fell_back_to_serial = True
        report.fallback_reason = f"{type(error).__name__}: {error}"
        emit(
            f"worker pool failed ({report.fallback_reason}); "
            f"re-running {len(tasks)} task(s) serially in task order"
        )
        report.used_workers = 1
        report.per_task_seconds = []
        report.worker_seconds = 0.0
        if initializer is not None:
            initializer(*initargs)
        results = run_serially()
        report.wall_seconds = time.perf_counter() - started
        return results, report

    aggregation_mark = time.perf_counter()
    report.per_task_seconds = seconds_by_index
    report.worker_seconds = float(sum(seconds_by_index))
    results = list(results_by_index)
    report.aggregation_seconds = time.perf_counter() - aggregation_mark
    report.wall_seconds = time.perf_counter() - started
    return results, report
