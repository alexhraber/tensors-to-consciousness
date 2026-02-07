from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_BRANCH = "main"
DEFAULT_API_IPS = ("140.82.114.6", "140.82.113.6", "140.82.112.6")


class SubmitPrError(RuntimeError):
    pass


@dataclass(frozen=True)
class RepoRef:
    owner: str
    name: str


def _run(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, cwd=ROOT, text=True).strip()


def _current_branch() -> str:
    branch = _run(["git", "branch", "--show-current"])
    if not branch:
        raise SubmitPrError("Cannot determine current branch (detached HEAD).")
    if branch in {"main", "master"}:
        raise SubmitPrError(f"Refusing to open PR from protected branch '{branch}'.")
    return branch


def _parse_repo_from_remote_url(remote_url: str) -> RepoRef:
    normalized = remote_url.strip()
    if normalized.startswith("git@github.com:"):
        normalized = normalized[len("git@github.com:") :]
    elif normalized.startswith("ssh://git@github.com/"):
        normalized = normalized[len("ssh://git@github.com/") :]
    elif normalized.startswith("https://github.com/"):
        normalized = normalized[len("https://github.com/") :]
    else:
        raise SubmitPrError(f"Unsupported origin URL format: {remote_url}")
    if normalized.endswith(".git"):
        normalized = normalized[:-4]
    parts = normalized.split("/")
    if len(parts) != 2 or not all(parts):
        raise SubmitPrError(f"Could not parse owner/repo from origin URL: {remote_url}")
    return RepoRef(owner=parts[0], name=parts[1])


def _origin_repo() -> RepoRef:
    remote_url = _run(["git", "remote", "get-url", "origin"])
    return _parse_repo_from_remote_url(remote_url)


def _gh_token() -> str:
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if token:
        return token
    try:
        token = _run(["gh", "auth", "token"]).strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise SubmitPrError(
            "No GitHub token available. Set GITHUB_TOKEN or run `gh auth login`."
        ) from exc
    if not token:
        raise SubmitPrError("No GitHub token available from `gh auth token`.")
    return token


def _api_resolve_ips() -> tuple[str, ...]:
    raw = os.environ.get("GH_API_RESOLVE_IPS", "").strip()
    if not raw:
        return DEFAULT_API_IPS
    items = tuple(part.strip() for part in raw.split(",") if part.strip())
    return items or DEFAULT_API_IPS


def _curl_api(
    method: str,
    path: str,
    token: str,
    data: dict[str, object] | None = None,
) -> object:
    headers = [
        "-H",
        f"Authorization: Bearer {token}",
        "-H",
        "Accept: application/vnd.github+json",
    ]
    for ip in _api_resolve_ips():
        cmd = [
            "curl",
            "-sS",
            "--fail-with-body",
            "--resolve",
            f"api.github.com:443:{ip}",
            *headers,
            "-X",
            method,
            f"https://api.github.com{path}",
        ]
        if data is not None:
            cmd.extend(["-d", json.dumps(data)])
        result = subprocess.run(
            cmd,
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode == 0:
            return json.loads(result.stdout or "{}")
    error = result.stderr.strip() if "result" in locals() else "no API attempts were made"
    raise SubmitPrError(f"GitHub API request failed for {path}: {error}")


def _find_existing_pr(owner: str, repo: str, head_branch: str, token: str) -> str | None:
    payload = _curl_api(
        "GET",
        f"/repos/{owner}/{repo}/pulls?state=open&head={owner}:{head_branch}",
        token,
    )
    if not isinstance(payload, list):
        return None
    if not payload:
        return None
    first = payload[0]
    if isinstance(first, dict):
        url = first.get("html_url")
        if isinstance(url, str) and url:
            return url
    return None


def _create_pr(owner: str, repo: str, head: str, base: str, token: str) -> str:
    body = {
        "title": "",
        "head": head,
        "base": base,
        "body": "",
        "maintainer_can_modify": True,
    }
    # Ask gh to generate the same title/body shape as --fill and parse output.
    try:
        title = _run(["gh", "pr", "view", "--json", "title", "--head", head])
        data = json.loads(title)
        if isinstance(data, dict):
            body["title"] = str(data.get("title") or "")
    except Exception:
        pass
    if not body["title"]:
        subject = _run(["git", "log", "-1", "--pretty=%s"])
        body["title"] = subject or f"chore: update {head}"
    response = _curl_api("POST", f"/repos/{owner}/{repo}/pulls", token, body)
    if not isinstance(response, dict):
        raise SubmitPrError("GitHub API response shape was invalid for PR create.")
    url = response.get("html_url")
    if not isinstance(url, str) or not url:
        raise SubmitPrError("GitHub API did not return a PR URL.")
    return url


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Push current branch and create (or reuse) PR with DNS-resolved GitHub API."
    )
    parser.add_argument("--base", default=DEFAULT_BASE_BRANCH, help="Base branch (default: main).")
    parser.add_argument(
        "--skip-push",
        action="store_true",
        help="Skip git push and only create/find PR.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    branch = _current_branch()
    if not args.skip_push:
        subprocess.run(["git", "push", "-u", "origin", branch], cwd=ROOT, check=True)
    repo = _origin_repo()
    token = _gh_token()
    existing = _find_existing_pr(repo.owner, repo.name, branch, token)
    if existing:
        print(existing)
        return 0
    pr_url = _create_pr(repo.owner, repo.name, branch, args.base, token)
    print(pr_url)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SubmitPrError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
