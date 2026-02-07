use anyhow::{anyhow, Context, Result};
use clap::{Args, Subcommand};
use regex::Regex;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::{mpsc, Arc, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};

#[derive(Subcommand)]
pub enum OpsCmd {
    /// Enforce repository git workflow policy (branch-first, naming)
    GitPolicy(GitPolicyArgs),
    /// Install repository Git hooks by setting core.hooksPath to .githooks
    InstallHooks,
    /// Bootstrap contributor toolchain (hooks + pre-commit)
    Bootstrap(BootstrapArgs),
    /// Push current branch and create/reuse PR via GitHub API (with optional DNS override)
    SubmitPr(SubmitPrArgs),
    /// Path-selected local CI gate (hook -> Docker -> CI-equivalent commands)
    PrePushGate(PrePushGateArgs),
}

#[derive(Args)]
pub struct GitPolicyArgs {
    #[arg(long, value_parser = ["pre-commit", "pre-push"])]
    pub hook: String,
}

#[derive(Args)]
pub struct BootstrapArgs {
    #[arg(long)]
    pub quiet: bool,
    #[arg(long)]
    pub strict: bool,
}

#[derive(Args)]
pub struct SubmitPrArgs {
    #[arg(long, default_value = "main")]
    pub base: String,
    #[arg(long)]
    pub skip_push: bool,
}

#[derive(Args)]
pub struct PrePushGateArgs {
    #[arg(long, default_value = "pre-push", value_parser = ["pre-commit", "pre-push"])]
    pub mode: String,
    #[arg(long)]
    pub no_cache: bool,
    #[arg(long)]
    pub jobs: Option<String>,
}

pub fn run_ops(cmd: OpsCmd) -> Result<()> {
    match cmd {
        OpsCmd::GitPolicy(args) => git_policy(&args.hook),
        OpsCmd::InstallHooks => install_hooks(),
        OpsCmd::Bootstrap(args) => bootstrap(args.quiet, args.strict),
        OpsCmd::SubmitPr(args) => submit_pr(&args.base, args.skip_push),
        OpsCmd::PrePushGate(args) => pre_push_gate(&args.mode, args.no_cache, args.jobs.as_deref()),
    }
}

fn repo_root() -> Result<PathBuf> {
    let out = Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .output()
        .context("git rev-parse --show-toplevel")?;
    if !out.status.success() {
        return Err(anyhow!(
            "failed to resolve repo root (are you in a git repo?)"
        ));
    }
    Ok(PathBuf::from(String::from_utf8_lossy(&out.stdout).trim()))
}

fn run_output(args: &[&str]) -> Result<String> {
    let root = repo_root()?;
    let out = Command::new(args[0])
        .args(&args[1..])
        .current_dir(root)
        .output()
        .with_context(|| format!("failed to run: {}", args.join(" ")))?;
    if !out.status.success() {
        return Err(anyhow!(
            "command failed ({}): {}",
            out.status,
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn run_status(args: &[&str], quiet: bool) -> Result<()> {
    let root = repo_root()?;
    if !quiet {
        eprintln!("+ {}", args.join(" "));
    }
    let status = Command::new(args[0])
        .args(&args[1..])
        .current_dir(root)
        .status()
        .with_context(|| format!("failed to run: {}", args.join(" ")))?;
    if !status.success() {
        return Err(anyhow!("command failed: {}", args.join(" ")));
    }
    Ok(())
}

fn install_hooks() -> Result<()> {
    let root = repo_root()?;
    let hooks = root.join(".githooks");
    if !hooks.exists() {
        return Err(anyhow!("missing hooks directory: {}", hooks.display()));
    }
    run_status(
        &[
            "git",
            "config",
            "core.hooksPath",
            hooks.to_string_lossy().as_ref(),
        ],
        false,
    )?;
    println!("Installed hooks path: {}", hooks.display());
    Ok(())
}

fn git_policy(hook: &str) -> Result<()> {
    let allowed_types = [
        "build", "chore", "ci", "docs", "feat", "fix", "perf", "refactor", "revert", "style",
        "test",
    ];
    let branch_pattern = Regex::new(
        r"^(?:build|chore|ci|docs|feat|fix|perf|refactor|revert|style|test)/[a-z0-9]+(?:-[a-z0-9]+)*$",
    )?;
    let branch = run_output(&["git", "rev-parse", "--abbrev-ref", "HEAD"])?;

    if branch == "main" || branch == "master" {
        match hook {
            "pre-commit" => {
                if std::env::var("ALLOW_MAIN_COMMIT").ok().as_deref() == Some("1") {
                    return Ok(());
                }
                return Err(anyhow!(
                    "Direct commits to main/master are blocked. Create a feature branch or set ALLOW_MAIN_COMMIT=1 for exceptional maintenance."
                ));
            }
            "pre-push" => {
                if std::env::var("ALLOW_MAIN_PUSH").ok().as_deref() == Some("1") {
                    return Ok(());
                }
                return Err(anyhow!(
                    "Direct pushes to main/master are blocked. Push a feature branch and open a PR, or set ALLOW_MAIN_PUSH=1 for exceptional maintenance."
                ));
            }
            _ => {}
        }
    }

    if branch != "HEAD" && branch != "main" && branch != "master" && !branch_pattern.is_match(&branch) {
        return Err(anyhow!(
            "Branch name must match 'type/scope-short-topic' with lowercase kebab-case.\nAllowed types: {}\nExamples: fix/ci-act-container-collision, docs/readme-minimal-refresh",
            allowed_types.join(", ")
        ));
    }
    Ok(())
}

fn which(cmd: &str) -> bool {
    std::env::var_os("PATH").is_some_and(|paths| {
        std::env::split_paths(&paths).any(|p| {
            let full = p.join(cmd);
            full.exists()
        })
    })
}

fn hooks_path_configured() -> bool {
    let root = match repo_root() {
        Ok(r) => r,
        Err(_) => return false,
    };
    let hooks = root.join(".githooks");
    let Ok(configured) = run_output(&["git", "config", "--get", "core.hooksPath"]) else {
        return false;
    };
    if configured.trim().is_empty() {
        return false;
    }
    let configured_path = PathBuf::from(configured);
    let resolved = if configured_path.is_absolute() {
        configured_path
    } else {
        root.join(configured_path)
    };
    resolved.canonicalize().ok() == hooks.canonicalize().ok()
}

fn bootstrap(quiet: bool, strict: bool) -> Result<()> {
    if hooks_path_configured() && which("pre-commit") {
        return Ok(());
    }
    let _ = install_hooks();

    // Ensure pre-commit is installed. Prefer `uv` when available.
    if !which("pre-commit") {
        if which("uv") {
            let root = repo_root()?;
            let status = Command::new("uv")
                .args(["pip", "install", "pre-commit"])
                // Keep cache local to the repo to avoid permission pitfalls and improve reuse.
                .env("UV_CACHE_DIR", root.join(".uv-cache"))
                .current_dir(&root)
                .status()
                .context("uv pip install pre-commit")?;
            if !status.success() && !quiet {
                eprintln!("warning: failed to install pre-commit via uv");
            }
        } else if which("python") {
            let _ = run_status(&["python", "-m", "pip", "install", "pre-commit"], quiet);
        }
    }

    if !which("pre-commit") {
        if strict {
            return Err(anyhow!(
                "pre-commit could not be installed automatically. Install it manually."
            ));
        }
        eprintln!("warning: pre-commit could not be installed automatically.");
        return Ok(());
    }
    let _ = run_status(&["pre-commit", "install", "--install-hooks"], quiet);
    Ok(())
}

#[derive(Clone, Debug)]
struct RepoRef {
    owner: String,
    name: String,
}

fn parse_repo_from_origin(remote_url: &str) -> Result<RepoRef> {
    let mut s = remote_url.trim().to_string();
    if let Some(rest) = s.strip_prefix("git@github.com:") {
        s = rest.to_string();
    } else if let Some(rest) = s.strip_prefix("ssh://git@github.com/") {
        s = rest.to_string();
    } else if let Some(rest) = s.strip_prefix("https://github.com/") {
        s = rest.to_string();
    } else {
        return Err(anyhow!("unsupported origin URL format: {remote_url}"));
    }
    if let Some(rest) = s.strip_suffix(".git") {
        s = rest.to_string();
    }
    let parts: Vec<&str> = s.split('/').collect();
    if parts.len() != 2 || parts[0].is_empty() || parts[1].is_empty() {
        return Err(anyhow!("could not parse owner/repo from origin URL: {remote_url}"));
    }
    Ok(RepoRef {
        owner: parts[0].to_string(),
        name: parts[1].to_string(),
    })
}

fn default_api_ips() -> Vec<String> {
    vec![
        "140.82.114.6".to_string(),
        "140.82.113.6".to_string(),
        "140.82.112.6".to_string(),
    ]
}

fn api_resolve_ips() -> Vec<String> {
    let raw = std::env::var("GH_API_RESOLVE_IPS").unwrap_or_default();
    let items: Vec<String> = raw
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect();
    if items.is_empty() {
        default_api_ips()
    } else {
        items
    }
}

fn gh_token() -> Result<String> {
    if let Ok(t) = std::env::var("GITHUB_TOKEN") {
        let t = t.trim().to_string();
        if !t.is_empty() {
            return Ok(t);
        }
    }
    // Fall back to `gh auth token`.
    let out = Command::new("gh")
        .args(["auth", "token"])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .context("failed to run `gh auth token`")?;
    if !out.status.success() {
        return Err(anyhow!(
            "No GitHub token available. Set GITHUB_TOKEN or run `gh auth login`."
        ));
    }
    let token = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if token.is_empty() {
        return Err(anyhow!("No GitHub token available from `gh auth token`."));
    }
    Ok(token)
}

fn curl_api(method: &str, path: &str, token: &str, data: Option<serde_json::Value>) -> Result<serde_json::Value> {
    let headers = [
        "-H",
        &format!("Authorization: Bearer {token}"),
        "-H",
        "Accept: application/vnd.github+json",
    ];
    let mut last_err: Option<String> = None;
    for ip in api_resolve_ips() {
        let mut cmd = Command::new("curl");
        cmd.args([
            "-sS",
            "--fail-with-body",
            "--resolve",
            &format!("api.github.com:443:{ip}"),
            headers[0],
            headers[1],
            headers[2],
            headers[3],
            "-X",
            method,
            &format!("https://api.github.com{path}"),
        ]);
        if let Some(ref d) = data {
            cmd.args(["-d", &d.to_string()]);
        }
        let out = cmd.output().context("failed to run curl")?;
        if out.status.success() {
            let raw = String::from_utf8_lossy(&out.stdout);
            let v: serde_json::Value = serde_json::from_str(raw.trim()).unwrap_or_else(|_| json!({}));
            return Ok(v);
        }
        last_err = Some(String::from_utf8_lossy(&out.stderr).trim().to_string());
    }
    Err(anyhow!(
        "GitHub API request failed for {path}: {}",
        last_err.unwrap_or_else(|| "no API attempts were made".to_string())
    ))
}

fn submit_pr(base: &str, skip_push: bool) -> Result<()> {
    let branch = run_output(&["git", "branch", "--show-current"])?;
    if branch.trim().is_empty() {
        return Err(anyhow!("Cannot determine current branch (detached HEAD)."));
    }
    if branch == "main" || branch == "master" {
        return Err(anyhow!("Refusing to open PR from protected branch '{branch}'"));
    }
    if !skip_push {
        let mut cmd = Command::new("git");
        cmd.args(["push", "-u", "origin", &branch]);
        // Optional DNS/IP override for SSH pushes when the local resolver is flaky.
        // If the user already set GIT_SSH_COMMAND, respect it.
        if std::env::var("GIT_SSH_COMMAND").ok().as_deref().unwrap_or("").trim().is_empty() {
            if let Ok(ip) = std::env::var("GH_SSH_HOST_IP") {
                let ip = ip.trim();
                if !ip.is_empty() {
                    cmd.env(
                        "GIT_SSH_COMMAND",
                        format!("ssh -o HostName={ip} -o HostKeyAlias=github.com -o BatchMode=yes"),
                    );
                }
            }
        }
        let status = cmd.status().context("git push")?;
        if !status.success() {
            return Err(anyhow!("git push failed for branch {branch}"));
        }
    }
    let origin = run_output(&["git", "remote", "get-url", "origin"])?;
    let repo = parse_repo_from_origin(&origin)?;
    let token = gh_token()?;

    // Find existing PR.
    let pulls = curl_api(
        "GET",
        &format!(
            "/repos/{}/{}/pulls?state=open&head={}:{}",
            repo.owner, repo.name, repo.owner, branch
        ),
        &token,
        None,
    )?;
    if let Some(url) = pulls
        .as_array()
        .and_then(|arr| arr.first())
        .and_then(|v| v.get("html_url"))
        .and_then(|v| v.as_str())
    {
        println!("{url}");
        return Ok(());
    }

    let subject = run_output(&["git", "log", "-1", "--pretty=%s"]).unwrap_or_else(|_| branch.clone());
    let body = json!({
      "title": subject,
      "head": branch,
      "base": base,
      "body": "",
      "maintainer_can_modify": true,
    });
    let resp = curl_api(
        "POST",
        &format!("/repos/{}/{}/pulls", repo.owner, repo.name),
        &token,
        Some(body),
    )?;
    let url = resp
        .get("html_url")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("GitHub API did not return a PR URL"))?;
    println!("{url}");
    Ok(())
}

// --------------------------
// Local CI gate
// --------------------------

fn cache_file(root: &Path) -> PathBuf {
    root.join(".git").join("explorer-cache").join("ci-gate.json")
}

fn best_base_ref() -> String {
    // Prefer upstream ref if configured.
    if let Ok(upstream) = run_output(&[
        "git",
        "rev-parse",
        "--abbrev-ref",
        "--symbolic-full-name",
        "@{upstream}",
    ]) {
        if !upstream.is_empty() {
            return upstream;
        }
    }
    if run_output(&["git", "rev-parse", "--verify", "origin/main"]).is_ok() {
        "origin/main".to_string()
    } else {
        "HEAD~1".to_string()
    }
}

fn changed_files_pre_push(base_ref: &str) -> Vec<String> {
    let diff = run_output(&["git", "diff", "--name-only", "--diff-filter=ACMR", &format!("{base_ref}...HEAD")]);
    diff.ok()
        .map(|s| s.lines().filter(|l| !l.trim().is_empty()).map(|l| l.to_string()).collect())
        .unwrap_or_default()
}

fn changed_files_pre_commit() -> Vec<String> {
    let diff = run_output(&["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"]);
    diff.ok()
        .map(|s| s.lines().filter(|l| !l.trim().is_empty()).map(|l| l.to_string()).collect())
        .unwrap_or_default()
}

fn is_lightweight_docs_only(paths: &[String]) -> bool {
    if paths.is_empty() {
        return false;
    }
    paths.iter().all(|p| p.ends_with(".md") || p.starts_with("docs/"))
}

fn has_prefix(paths: &[String], prefixes: &[&str]) -> bool {
    paths.iter().any(|p| prefixes.iter().any(|pre| p.starts_with(pre)))
}

fn has_exact(paths: &[String], names: &[&str]) -> bool {
    let set: BTreeSet<&str> = names.iter().copied().collect();
    paths.iter().any(|p| set.contains(p.as_str()))
}

fn select_ci_tasks(paths: &[String]) -> Vec<String> {
    if is_lightweight_docs_only(paths) {
        return vec![];
    }

    let test_inputs = has_exact(
        paths,
        &[
            "explorer.py",
            ".python-version",
            "mise.toml",
            "Cargo.toml",
            "Cargo.lock",
            ".github/workflows/ci.yml",
            ".github/ci/requirements-test.txt",
        ],
    ) || has_prefix(paths, &["transforms/", "frameworks/", "tools/", "tests/", "crates/"]);

    let transform_contract_inputs = has_prefix(paths, &["transforms/"])
        || has_exact(
            paths,
            &[
                "frameworks/engine.py",
                "tools/runtime.py",
                "tools/playground.py",
                "mise.toml",
                ".github/workflows/ci.yml",
            ],
        );

    let framework_contract_inputs = has_prefix(paths, &["frameworks/"])
        || has_exact(
            paths,
            &["tools/setup.py", "tools/runtime.py", "mise.toml", ".github/workflows/ci.yml"],
        );

    let mut tasks: Vec<String> = vec![];
    if test_inputs {
        tasks.push("ci-test".to_string());
    }
    if transform_contract_inputs {
        tasks.push("ci-transform-contract".to_string());
    }
    if framework_contract_inputs {
        tasks.push("ci-framework-contract-jax".to_string());
    }
    tasks
}

fn sha256_hex(s: &str) -> String {
    let mut h = Sha256::new();
    h.update(s.as_bytes());
    hex::encode(h.finalize())
}

fn cache_signature(mode: &str, paths: &[String]) -> Result<String> {
    let head = run_output(&["git", "rev-parse", "HEAD"])?;
    let payload = if mode == "pre-push" {
        format!(
            "mode={mode}\nhead={head}\npaths={}",
            paths.iter().cloned().collect::<Vec<_>>().join("\n")
        )
    } else {
        let staged = run_output(&["git", "diff", "--cached"]).unwrap_or_default();
        format!("mode={mode}\nhead={head}\nstaged_diff={staged}")
    };
    Ok(sha256_hex(&payload))
}

fn now_ts() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

fn load_cache(p: &Path) -> BTreeMap<String, f64> {
    let raw = fs::read_to_string(p).ok();
    let Some(raw) = raw else { return BTreeMap::new(); };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&raw) else {
        return BTreeMap::new();
    };
    let mut out = BTreeMap::new();
    if let Some(entries) = v.get("entries").and_then(|x| x.as_object()) {
        for (k, vv) in entries {
            let ts = vv.get("ts").and_then(|t| t.as_f64()).unwrap_or(0.0);
            out.insert(k.clone(), ts);
        }
    }
    out
}

fn save_cache(p: &Path, entries: &BTreeMap<String, f64>) -> Result<()> {
    if let Some(dir) = p.parent() {
        fs::create_dir_all(dir).ok();
    }
    let mut obj = serde_json::Map::new();
    for (k, ts) in entries {
        obj.insert(k.clone(), json!({ "ts": ts }));
    }
    let payload = json!({ "entries": obj });
    fs::write(p, serde_json::to_string_pretty(&payload)?).context("write cache")?;
    Ok(())
}

fn parse_jobs_value(raw: Option<&str>) -> Option<usize> {
    let raw = raw?.trim().to_lowercase();
    if raw.is_empty() {
        return None;
    }
    if matches!(raw.as_str(), "nproc" | "cpu" | "cpus" | "max") {
        return Some(std::thread::available_parallelism().ok().map(|n| n.get()).unwrap_or(2));
    }
    raw.parse::<usize>().ok()
}

fn ensure_docker_available() -> Result<()> {
    let out = Command::new("docker")
        .args(["version"])
        .output()
        .context("docker version")?;
    if !out.status.success() {
        return Err(anyhow!("docker is required for local CI gating"));
    }
    Ok(())
}

fn ensure_ci_image(tag: &str) -> Result<()> {
    // Rely on the Docker layer cache. The pre-push cache decides whether we need to run at all.
    let status = Command::new("docker")
        .args(["build", "--target", "ci", "-t", tag, "."])
        .status()
        .context("docker build (ci target)")?;
    if !status.success() {
        return Err(anyhow!("failed to build ci image: {tag}"));
    }
    Ok(())
}

fn run_ci_task(task: &str, image: &str) -> Result<()> {
    let script = match task {
        "ci-test" => {
            r#"git config --global --add safe.directory /workspace && \
python -m py_compile $(git ls-files "*.py") && \
python -m coverage erase && \
python -m coverage run -m tests --suite unit && \
python -m coverage run --append -m tests --suite integration && \
python -m coverage report --show-missing --fail-under=69 && \
python -m coverage xml -o coverage.xml && \
explorer --help >/dev/null && \
explorer list-transforms >/dev/null"#
        }
        "ci-docs-sync" => r#"git config --global --add safe.directory /workspace && python tools/generate_catalog_docs.py && git diff --exit-code -- docs/reference/transforms.md docs/reference/frameworks.md"#,
        "ci-transform-contract" => r#"python - <<'PY'
from transforms.catalog import catalog_framework_interface
from transforms.definitions import TRANSFORM_DEFINITIONS
from transforms.registry import TRANSFORM_MAP
import sys

registry_keys = set(TRANSFORM_MAP.keys())
definition_keys = set(TRANSFORM_DEFINITIONS.keys())

missing_defs = sorted(registry_keys - definition_keys)
extra_defs = sorted(definition_keys - registry_keys)

contract = catalog_framework_interface()
missing_contract = []
if not contract.get("utils_entrypoints"):
    missing_contract.append("utils_entrypoints")
if not contract.get("ops_adapter"):
    missing_contract.append("ops_adapter")

if missing_defs or extra_defs:
    print("Transform contract violations:")
    if missing_defs:
        print(f" - missing definitions: {missing_defs}")
    if extra_defs:
        print(f" - unregistered definitions: {extra_defs}")
    sys.exit(1)
if missing_contract:
    print("Transform interface contract is incomplete:")
    for key in missing_contract:
        print(f" - missing: {key}")
    sys.exit(1)
print("Transform contract checks passed.")
PY"#,
        "ci-framework-contract-jax" => r#"FRAMEWORK=jax python - <<'PY'
import os
import pathlib
import sys

fw = os.environ["FRAMEWORK"]
root = pathlib.Path("frameworks") / fw
if not root.exists():
    print(f"Missing framework directory: {root}")
    sys.exit(1)

required = ["utils.py", "test_setup.py"]
missing = [name for name in required if not (root / name).exists()]
if missing:
    print(f"{fw}: missing required files:")
    for name in missing:
        print(f" - {root / name}")
    sys.exit(1)

transforms_backend = root / "transforms"
if not transforms_backend.exists():
    print(f"{fw}: missing transforms adapter directory: {transforms_backend}")
    sys.exit(1)

print(f"{fw}: framework contract passed.")
PY"#,
        other => return Err(anyhow!("unknown ci task: {other}")),
    };

    // Git doesn't expand $PWD inside args; rely on the shell and keep the workdir stable.
    // We execute docker from the repo root (callers should chdir there already).
    let status = Command::new("bash")
        .args(["-lc", &format!("docker run --rm -v \"$PWD:/workspace\" -w /workspace {image} bash -lc {q}", q = shell_quote(script))])
        .status()
        .with_context(|| format!("run ci task: {task}"))?;
    if !status.success() {
        return Err(anyhow!("ci task failed: {task}"));
    }
    Ok(())
}

fn shell_quote(s: &str) -> String {
    // Minimal single-quote wrapper safe for bash -lc.
    let mut out = String::with_capacity(s.len() + 2);
    out.push('\'');
    for ch in s.chars() {
        if ch == '\'' {
            out.push_str("'\\''");
        } else {
            out.push(ch);
        }
    }
    out.push('\'');
    out
}

fn resolve_workers(jobs: Option<&str>, task_count: usize) -> usize {
    if task_count <= 1 {
        return 1;
    }
    let parsed = parse_jobs_value(jobs).or_else(|| parse_jobs_value(std::env::var("CI_GATE_JOBS").ok().as_deref()));
    let n = parsed.unwrap_or_else(|| std::thread::available_parallelism().ok().map(|n| n.get()).unwrap_or(2));
    n.clamp(1, task_count)
}

fn pre_push_gate(mode: &str, no_cache: bool, jobs: Option<&str>) -> Result<()> {
    let root = repo_root()?;
    std::env::set_current_dir(&root).context("chdir to repo root")?;
    let cache_disabled = no_cache || std::env::var("CI_GATE_NO_CACHE").ok().as_deref() == Some("1");
    let base_ref = best_base_ref();
    let paths = if mode == "pre-commit" {
        changed_files_pre_commit()
    } else {
        changed_files_pre_push(&base_ref)
    };
    let tasks = select_ci_tasks(&paths);
    if tasks.is_empty() {
        if mode == "pre-commit" {
            eprintln!("[pre-commit] No CI-relevant staged changes; skipping CI jobs.");
        } else {
            eprintln!("[pre-push] No CI-relevant changes since {base_ref}; skipping CI jobs.");
        }
        return Ok(());
    }

    let signature = cache_signature(mode, &paths)?;
    let cache_path = cache_file(&root);
    let mut cache_entries = if cache_disabled { BTreeMap::new() } else { load_cache(&cache_path) };

    let prefix = if mode == "pre-commit" { "pre-commit" } else { "pre-push" };
    if mode == "pre-push" {
        eprintln!("[{prefix}] Base ref: {base_ref}");
    }
    eprintln!("[{prefix}] Changed files: {}", paths.len());

    let mut runnable: Vec<(String, String)> = vec![];
    for task in tasks {
        let cache_key = format!("{mode}:{task}:{signature}");
        if !cache_disabled && cache_entries.contains_key(&cache_key) {
            eprintln!("[{prefix}] Skipping {task} (cached success).");
            continue;
        }
        runnable.push((task, cache_key));
    }
    if runnable.is_empty() {
        eprintln!("[{prefix}] Completed selected CI jobs.");
        return Ok(());
    }

    ensure_docker_available()?;
    let image = "explorer-ci:local";
    ensure_ci_image(image)?;

    let workers = resolve_workers(jobs, runnable.len());
    eprintln!(
        "[{prefix}] Running {} CI job(s) with {workers} worker(s).",
        runnable.len()
    );

    let queue = Arc::new(Mutex::new(runnable));
    let (tx, rx) = mpsc::channel::<Result<(String, String)>>();
    for _ in 0..workers {
        let queue = Arc::clone(&queue);
        let tx = tx.clone();
        let image = image.to_string();
        std::thread::spawn(move || loop {
            let next = {
                let mut q = queue.lock().unwrap();
                q.pop()
            };
            let Some((task, cache_key)) = next else { break };
            let r = run_ci_task(&task, &image).map(|_| (task, cache_key));
            let _ = tx.send(r);
        });
    }
    drop(tx);

    for r in rx {
        let (task, cache_key) = r?;
        eprintln!("[{prefix}] Completed {task}");
        if !cache_disabled {
            cache_entries.insert(cache_key, now_ts());
        }
    }
    if !cache_disabled {
        let _ = save_cache(&cache_path, &cache_entries);
    }
    eprintln!("[{prefix}] Completed selected CI jobs.");
    Ok(())
}
