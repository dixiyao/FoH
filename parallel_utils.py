"""Small, deterministic thread-pool helpers for API-bound benchmark clients."""

from __future__ import annotations

import multiprocessing
import queue
import traceback
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Callable, Dict, Iterable, Iterator, List, Tuple, TypeVar


T = TypeVar("T")
R = TypeVar("R")


DEFAULT_NUM_WORKERS = 8


def worker_count(value: int) -> int:
    """Validate and normalize a CLI worker count."""
    count = int(value)
    if count < 1:
        raise ValueError("num_workers must be at least 1")
    return count


def parallel_map_ordered(
    function: Callable[[T], R],
    items: Iterable[T],
    *,
    num_workers: int = DEFAULT_NUM_WORKERS,
) -> List[R]:
    """Run independent API-bound calls concurrently and retain input ordering."""
    values = list(items)
    workers = min(worker_count(num_workers), len(values)) if values else 1
    if workers == 1:
        return [function(item) for item in values]
    # Keep at most one wave in flight. pool.map queues the entire experiment
    # and observes errors in input order, so a fatal error in a later worker
    # could otherwise launch every remaining case before reaching the caller.
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="task-client")
    pending = {}
    results: Dict[int, R] = {}
    next_index = 0
    try:
        while next_index < len(values) or pending:
            while next_index < len(values) and len(pending) < workers:
                pending[pool.submit(function, values[next_index])] = next_index
                next_index += 1
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            # Check every completed future for an exception before refilling.
            for future in done:
                results[pending.pop(future)] = future.result()
        return [results[index] for index in range(len(values))]
    finally:
        for future in pending:
            future.cancel()
        # Already-running cases must finish cleanup before releasing the
        # experiment lock. No further cases are submitted after an error.
        pool.shutdown(wait=True, cancel_futures=True)


def parallel_map_process_completed(
    function: Callable[[T], R],
    items: Iterable[T],
    *,
    num_workers: int = DEFAULT_NUM_WORKERS,
) -> Iterator[Tuple[T, R]]:
    """Run work in forked processes and yield results in completion order.

    Fork is intentional: benchmark lifecycle callables close over pipeline and
    task state that cannot be serialized for spawn-based process pools. Linux
    Slurm workers and macOS both provide fork. Each child pulls multiple items
    from a shared queue, so at most ``num_workers`` processes exist at once.
    """
    values = list(items)
    workers = min(worker_count(num_workers), len(values)) if values else 1
    if workers == 1:
        for item in values:
            yield item, function(item)
        return

    if "fork" not in multiprocessing.get_all_start_methods():
        raise RuntimeError(
            "Process parallelism requires the multiprocessing 'fork' start method"
        )
    context = multiprocessing.get_context("fork")
    task_queue = context.Queue()
    result_queue = context.Queue()
    for index, item in enumerate(values):
        task_queue.put((index, item))
    for _ in range(workers):
        task_queue.put(None)

    def _worker() -> None:
        while True:
            queued = task_queue.get()
            if queued is None:
                return
            index, item = queued
            try:
                result_queue.put(("ok", index, function(item)))
            except BaseException as exc:  # noqa: BLE001
                result_queue.put(
                    (
                        "error",
                        index,
                        f"{type(exc).__name__}: {exc}",
                        traceback.format_exc(),
                    )
                )

    processes = [context.Process(target=_worker) for _ in range(workers)]
    for process in processes:
        process.start()

    received = 0
    try:
        while received < len(values):
            try:
                message = result_queue.get(timeout=1.0)
            except queue.Empty:
                if not any(process.is_alive() for process in processes):
                    raise RuntimeError(
                        f"Process workers exited after returning {received}/{len(values)} results"
                    )
                continue
            received += 1
            status, index, *payload = message
            if status == "error":
                error, child_traceback = payload
                raise RuntimeError(
                    f"Process worker failed for item {index}: {error}\n{child_traceback}"
                )
            yield values[index], payload[0]
    finally:
        for process in processes:
            process.join(timeout=2.0)
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()
        task_queue.close()
        result_queue.close()


def add_num_workers_argument(parser) -> None:
    parser.add_argument(
        "--num-workers",
        "--num_workers",
        "--num-worker",
        "--num_worker",
        dest="num_workers",
        type=worker_count,
        default=DEFAULT_NUM_WORKERS,
        help=(
            "Maximum number of independent API-backed worker processes to run "
            "concurrently within each task batch; also caps independent server "
            f"prompt-rewrite processes (default: {DEFAULT_NUM_WORKERS})."
        ),
    )
