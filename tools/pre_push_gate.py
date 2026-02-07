from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CACHE_FILE = ROOT / ".git" / "explorer-cache" / "act-gate.json"


def _git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()


def _best_base_ref() -> str:
    try:
        upstream = _git_output("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}")
        if upstream:
            return upstream
    except subprocess.CalledProcessError:
        pass
    try:
        _git_output("rev-parse", "--verify", "origin/main")
        return "origin/main"
    except subprocess.CalledProcessError:
        return "HEAD~1"


def _changed_files_for_pre_push(base_ref: str) -> list[str]:
    try:
        diff = _git_output("diff", "--name-only", "--diff-filter=ACMR", f"{base_ref}...HEAD")
        return [line for line in diff.splitlines() if line]
    except subprocess.CalledProcessError:
        return []


def _changed_files_for_pre_commit() -> list[str]:
    try:
        diff = _git_output("diff", "--cached", "--name-only", "--diff-filter=ACMR")
        return [line for line in diff.splitlines() if line]
    except subprocess.CalledProcessError:
        return []


def _has_prefix(paths: list[str], *prefixes: str) -> bool:
    return any(any(path.startswith(prefix) for prefix in prefixes) for path in paths)


def _has_exact(paths: list[str], *names: str) -> bool:
    names_set = set(names)
    return any(path in names_set for path in paths)


def _is_lightweight_docs_only(paths: list[str]) -> bool:
    if not paths:
        return False
    # Keep README/docs markdown edits fast: these do not need local act runs.
    return all(path.endswith(".md") or path.startswith("docs/") for path in paths)


def select_act_tasks(paths: list[str]) -> list[str]:
    # Keep local gate fast for docs-only edits (README/docs markdown).
    if _is_lightweight_docs_only(paths):
        return []

    # Mirror CI workflow path filters by job scope.
    test_inputs = _has_exact(
        paths,
        "explorer.py",
        ".python-version",
        "mise.toml",
        "Cargo.toml",
        "Cargo.lock",
        ".github/workflows/ci.yml",
    ) or _has_prefix(paths, "transforms/", "frameworks/", "tools/", "tests/", "crates/")
    test_inputs = test_inputs or _has_exact(paths, ".github/ci/requirements-test.txt", "tools/pre_push_gate.py")

    transform_contract_inputs = _has_prefix(paths, "transforms/") or _has_exact(
        paths,
        "frameworks/engine.py",
        "tools/runtime.py",
        "tools/playground.py",
        "mise.toml",
        ".github/workflows/ci.yml",
    )

    framework_contract_inputs = _has_prefix(paths, "frameworks/") or _has_exact(
        paths,
        "tools/setup.py",
        "tools/runtime.py",
        "mise.toml",
        ".github/workflows/ci.yml",
    )

    docs_sync_inputs = _has_prefix(paths, "docs/") or _has_exact(
        paths, "transforms/transforms.json", "tools/generate_catalog_docs.py", "tools/runtime.py", "mise.toml", ".github/workflows/ci.yml"
    )

    tasks: list[str] = []
    if test_inputs:
        tasks.append("act-ci-test")
    if transform_contract_inputs:
        tasks.append("act-ci-transform-contract")
    if framework_contract_inputs:
        tasks.append("act-ci-framework-contract-jax")
    if docs_sync_inputs and not _is_lightweight_docs_only(paths):
        tasks.append("act-ci-docs-sync")
    return tasks


def _run_task(task: str) -> None:
    env = os.environ.copy()
    # `act` will attempt to authenticate action clones if `GITHUB_TOKEN` is present.
    # For local gates we prefer anonymous clones (and avoid surprising failures from
    # stale/invalid tokens).
    if task.startswith("act-") or task.startswith("act-ci-"):
        env.pop("GITHUB_TOKEN", None)
    subprocess.run(["mise", "run", task], cwd=ROOT, check=True, env=env)


def _parse_jobs_value(raw: str | None) -> int | None:
    if raw is None:
        return None
    value = raw.strip().lower()
    if not value:
        return None
    if value in {"nproc", "cpu", "cpus", "max"}:
        return os.cpu_count() or 2
    try:
        return int(value)
    except ValueError:
        return None


def _resolve_jobs(jobs_arg: str | None, tasks: list[str]) -> int:
    task_count = len(tasks)
    if task_count <= 1:
        return 1
    parsed_jobs_arg = _parse_jobs_value(jobs_arg)
    if parsed_jobs_arg is not None:
        return max(1, min(parsed_jobs_arg, task_count))
    parsed_jobs_env = _parse_jobs_value(os.environ.get("CI_GATE_JOBS"))
    if parsed_jobs_env is not None:
        return max(1, min(parsed_jobs_env, task_count))
    cpu = os.cpu_count() or 2
    return max(1, min(cpu, task_count))


def _load_cache() -> dict[str, dict[str, float]]:
    try:
        payload = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        entries = payload.get("entries")
        if isinstance(entries, dict):
            normalized: dict[str, dict[str, float]] = {}
            for k, v in entries.items():
                if isinstance(k, str) and isinstance(v, dict):
                    ts = float(v.get("ts", 0.0))
                    normalized[k] = {"ts": ts}
            return normalized
    except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError, TypeError):
        pass
    return {}


def _save_cache(entries: dict[str, dict[str, float]]) -> None:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    if len(entries) > 400:
        newest = sorted(entries.items(), key=lambda item: item[1].get("ts", 0.0), reverse=True)[:200]
        entries = dict(newest)
    CACHE_FILE.write_text(json.dumps({"entries": entries}, indent=2, sort_keys=True), encoding="utf-8")


def _cache_signature(mode: str, paths: list[str]) -> str:
    head = _git_output("rev-parse", "HEAD")
    if mode == "pre-push":
        payload = f"mode={mode}\nhead={head}\npaths=" + "\n".join(sorted(paths))
    else:
        staged = _git_output("diff", "--cached")
        payload = f"mode={mode}\nhead={head}\nstaged_diff={staged}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run path-selected act CI jobs.")
    parser.add_argument(
        "--mode",
        choices=["pre-commit", "pre-push"],
        default="pre-push",
        help="Change scope mode: staged files for pre-commit or branch diff for pre-push.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable local gate cache and force all selected act jobs to run.",
    )
    parser.add_argument(
        "--jobs",
        type=str,
        default=None,
        help="Parallel local gate jobs (int or 'nproc').",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cache_disabled = args.no_cache or os.environ.get("CI_GATE_NO_CACHE") == "1"
    base_ref = _best_base_ref()
    paths = (
        _changed_files_for_pre_commit() if args.mode == "pre-commit" else _changed_files_for_pre_push(base_ref)
    )
    tasks = select_act_tasks(paths)
    if not tasks:
        if args.mode == "pre-commit":
            print("[pre-commit] No CI-relevant staged changes; skipping act jobs.")
        else:
            print(f"[pre-push] No CI-relevant changes since {base_ref}; skipping act jobs.")
        return 0

    prefix = "pre-commit" if args.mode == "pre-commit" else "pre-push"
    signature = _cache_signature(args.mode, paths)
    cache_entries = {} if cache_disabled else _load_cache()
    cache_dirty = False
    if args.mode == "pre-push":
        print(f"[{prefix}] Base ref: {base_ref}")
    print(f"[{prefix}] Changed files: {len(paths)}")
    runnable_tasks: list[tuple[str, str]] = []
    for task in tasks:
        cache_key = f"{args.mode}:{task}:{signature}"
        if not cache_disabled and cache_key in cache_entries:
            print(f"[{prefix}] Skipping {task} (cached success).")
            continue
        runnable_tasks.append((task, cache_key))

    if not runnable_tasks:
        print(f"[{prefix}] Completed selected act jobs.")
        return 0

    workers = _resolve_jobs(args.jobs, [task for task, _ in runnable_tasks])
    # Running multiple `act` invocations concurrently is unstable because act uses
    # deterministic container names for shared workflow jobs (e.g., "Change Detection"),
    # which can collide. Default to serial execution unless explicitly enabled.
    if workers > 1 and all(task.startswith("act-") or task.startswith("act-ci-") for task, _ in runnable_tasks):
        if os.environ.get("ALLOW_PARALLEL_ACT") != "1":
            workers = 1
    print(f"[{prefix}] Running {len(runnable_tasks)} act job(s) with {workers} worker(s).")

    failures: list[tuple[str, int]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {}
        for task, cache_key in runnable_tasks:
            print(f"[{prefix}] Running {task}")
            future = pool.submit(_run_task, task)
            future_map[future] = (task, cache_key)

        for future in as_completed(future_map):
            task, cache_key = future_map[future]
            try:
                future.result()
                if not cache_disabled:
                    cache_entries[cache_key] = {"ts": time.time()}
                    cache_dirty = True
            except subprocess.CalledProcessError as exc:
                failures.append((task, exc.returncode or 1))

    # `act` can collide on shared dependency container names when multiple jobs from
    # the same workflow are launched concurrently. If that happens, retry act tasks
    # once in serial mode to preserve a fast path while keeping pushes reliable.
    if failures and workers > 1:
        failed_task_names = {task for task, _ in failures}
        failed_non_act = [task for task, _ in failures if not task.startswith("act-ci-")]
        if not failed_non_act:
            print(f"[{prefix}] Retrying failed act jobs sequentially (act container-name collision fallback).")
            failures = []
            for task, cache_key in runnable_tasks:
                if task not in failed_task_names:
                    continue
                print(f"[{prefix}] Retrying {task}")
                try:
                    _run_task(task)
                    if not cache_disabled:
                        cache_entries[cache_key] = {"ts": time.time()}
                        cache_dirty = True
                except subprocess.CalledProcessError as exc:
                    failures.append((task, exc.returncode or 1))

    if failures:
        for task, code in failures:
            print(f"[{prefix}] ERROR {task} failed (exit {code}).", file=sys.stderr)
        print(
            f"[{prefix}] Install/refresh toolchain with `mise install`, "
            f"then retry. For one-off bypass: `SKIP_ACT=1 git commit` or `SKIP_ACT=1 git push`.",
            file=sys.stderr,
        )
        return failures[0][1]

    if not cache_disabled and cache_dirty:
        _save_cache(cache_entries)
    print(f"[{prefix}] Completed selected act jobs.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[ci-gate] Interrupted.", file=sys.stderr)
        raise SystemExit(130)
