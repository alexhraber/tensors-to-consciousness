use regex::Regex;
use std::{
    fs,
    path::{Path, PathBuf},
};

fn repo_root() -> PathBuf {
    // crates/explorer -> repo root
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .unwrap()
        .to_path_buf()
}

fn read_text(path: &Path) -> String {
    fs::read_to_string(path).unwrap_or_else(|e| panic!("failed reading {}: {e}", path.display()))
}

fn list_md_files(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let rd = fs::read_dir(&dir).unwrap_or_else(|e| panic!("read_dir {}: {e}", dir.display()));
        for entry in rd {
            let entry = entry.unwrap();
            let p = entry.path();
            if p.is_dir() {
                stack.push(p);
                continue;
            }
            if p.extension().and_then(|s| s.to_str()) == Some("md") {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

#[test]
fn contract_ci_workflow_contains_render_smoke_job() {
    let root = repo_root();
    let ci = root.join(".github/workflows/ci.yml");
    assert!(ci.exists(), "missing ci workflow: {}", ci.display());
    let text = read_text(&ci);
    for fragment in [
        "Build render image",
        "target: render",
        "Render smoke checks (container)",
    ] {
        assert!(
            text.contains(fragment),
            "ci.yml missing fragment: {fragment}"
        );
    }
}

#[test]
fn contract_mise_render_tasks_include_headless_capture_deps() {
    let root = repo_root();
    let mise = root.join("mise.toml");
    assert!(mise.exists(), "missing mise.toml");
    let text = read_text(&mise);
    for fragment in ["xvfb", "xterm", "xdotool", "--tui-capture headless"] {
        assert!(
            text.contains(fragment),
            "mise.toml missing fragment: {fragment}"
        );
    }
}

#[test]
fn contract_assets_sync_workflow_triggers_only_on_asset_inputs() {
    let root = repo_root();
    let wf = root.join(".github/workflows/assets-readme-sync.yml");
    assert!(wf.exists(), "missing workflow: {}", wf.display());
    let text = read_text(&wf);
    assert!(
        !text.contains("      - \"README.md\""),
        "assets workflow must not trigger on README-only edits"
    );
    for fragment in [
        "transforms/**",
        "frameworks/**",
        "tools/shinkei.py",
        "tools/input_controls.py",
        "tools/generate_render_assets.py",
        "examples/inputs.framework_matrix.json",
        "examples/inputs.spectral_sweep.json",
        ".python-version",
        "mise.toml",
        ".github/ci/requirements-test.txt",
        ".github/workflows/assets-readme-sync.yml",
    ] {
        assert!(
            text.contains(fragment),
            "assets workflow missing fragment: {fragment}"
        );
    }
}

#[test]
fn contract_docker_compose_exposes_interactive_explorer_services() {
    let root = repo_root();
    let compose = root.join("docker-compose.yml");
    assert!(compose.exists(), "missing docker-compose.yml");
    let text = read_text(&compose);
    for fragment in [
        "explorer:",
        "explorer-nvidia:",
        "explorer-amd:",
        "explorer-intel:",
        "explorer-apple:",
        "stdin_open: true",
        "tty: true",
        "command: [\"explorer\"]",
        "explorer_config:",
    ] {
        assert!(
            text.contains(fragment),
            "docker-compose.yml missing fragment: {fragment}"
        );
    }
}

#[test]
fn contract_hooks_and_mise_call_rust_ops() {
    let root = repo_root();
    let pre_commit = read_text(&root.join(".githooks/pre-commit"));
    let pre_push = read_text(&root.join(".githooks/pre-push"));
    assert!(
        pre_commit.contains("explorer ops git-policy --hook pre-commit"),
        "pre-commit hook must enforce rust git policy"
    );
    assert!(
        pre_commit.contains("explorer ops bootstrap"),
        "pre-commit hook must bootstrap toolchain"
    );
    assert!(
        pre_push.contains("explorer ops pre-push-gate"),
        "pre-push hook must invoke rust pre-push-gate"
    );

    let mise = read_text(&root.join("mise.toml"));
    assert!(
        mise.contains("explorer ops pre-push-gate"),
        "mise tasks must call rust pre-push gate"
    );
    assert!(
        mise.contains("explorer ops submit-pr"),
        "mise tasks must call rust submit-pr"
    );
}

#[test]
fn contract_markdown_relative_links_resolve() {
    let root = repo_root();
    let link_re = Regex::new(r"\[[^\]]+\]\(([^)]+)\)").unwrap();

    let mut md_files = vec![root.join("README.md")];
    md_files.extend(list_md_files(&root.join("docs")));

    for md in md_files {
        let text = read_text(&md);
        for cap in link_re.captures_iter(&text) {
            let raw = cap.get(1).unwrap().as_str().trim();
            if raw.is_empty()
                || raw.starts_with("http://")
                || raw.starts_with("https://")
                || raw.starts_with("mailto:")
                || raw.starts_with('#')
            {
                continue;
            }
            let target = raw.split('#').next().unwrap_or("").trim();
            if target.is_empty() {
                continue;
            }
            let resolved = md
                .parent()
                .unwrap_or(Path::new("."))
                .join(target)
                .canonicalize()
                .unwrap_or_else(|_| md.parent().unwrap().join(target));
            assert!(
                resolved.exists(),
                "broken link: source={} target={raw} resolved={}",
                md.display(),
                resolved.display()
            );
        }
    }
}

#[test]
fn contract_framework_layout_surface_exists() {
    let root = repo_root();
    let frameworks = ["mlx", "jax", "pytorch", "numpy", "keras", "cupy"];
    for fw in frameworks {
        let dir = root.join("frameworks").join(fw);
        assert!(dir.is_dir(), "missing framework dir: {}", dir.display());
        assert!(
            dir.join("utils.py").is_file(),
            "missing utils.py for {fw}"
        );
        assert!(
            dir.join("test_setup.py").is_file(),
            "missing test_setup.py for {fw}"
        );
        assert!(
            dir.join("transforms/__init__.py").is_file(),
            "missing transforms/__init__.py for {fw}"
        );

        // Lightweight export contract: function definitions must be present.
        let utils = read_text(&dir.join("utils.py"));
        for func in ["def normal", "def render_stage", "def _to_numpy"] {
            assert!(
                utils.contains(func),
                "{fw} utils.py missing required function: {func}"
            );
        }
        let setup = read_text(&dir.join("test_setup.py"));
        assert!(
            setup.contains("normal"),
            "{fw} test_setup.py should reference normal()"
        );
        assert!(
            setup.contains("Setup Test"),
            "{fw} test_setup.py should print setup banner"
        );
    }
}

