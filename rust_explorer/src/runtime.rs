use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveConfig {
    pub framework: String,
    pub venv: String,
}

pub fn default_framework_for_platform() -> &'static str {
    if cfg!(target_os = "macos") {
        "mlx"
    } else {
        "jax"
    }
}

pub fn repo_root() -> Result<PathBuf> {
    let cwd = std::env::current_dir().context("current_dir")?;
    Ok(cwd)
}

pub fn load_active_config(root: &Path) -> Option<ActiveConfig> {
    let p = root.join(".t2c").join("config.json");
    let raw = fs::read_to_string(p).ok()?;
    serde_json::from_str(&raw).ok()
}

pub fn venv_python(venv_dir: &Path) -> PathBuf {
    if cfg!(target_os = "windows") {
        venv_dir.join("Scripts").join("python.exe")
    } else {
        venv_dir.join("bin").join("python")
    }
}

pub fn default_venv_for_framework(framework: &str) -> PathBuf {
    PathBuf::from(format!(".venv-{framework}"))
}

#[derive(Debug, Clone)]
pub struct ResolvedRuntime {
    pub framework: String,
    pub venv_dir: PathBuf,
    pub engine_python: PathBuf,
}

pub fn resolve_runtime(
    root: &Path,
    framework_override: Option<&str>,
    venv_override: Option<&str>,
) -> ResolvedRuntime {
    let cfg = load_active_config(root);
    let framework = framework_override
        .map(|s| s.to_string())
        .or_else(|| cfg.as_ref().map(|c| c.framework.clone()))
        .unwrap_or_else(|| default_framework_for_platform().to_string());

    let venv_dir = if let Some(v) = venv_override {
        PathBuf::from(v)
    } else if let Some(c) = cfg.as_ref() {
        if c.framework == framework && !c.venv.trim().is_empty() {
            PathBuf::from(&c.venv)
        } else {
            default_venv_for_framework(&framework)
        }
    } else {
        default_venv_for_framework(&framework)
    };

    let engine_python = venv_python(&venv_dir);
    ResolvedRuntime {
        framework,
        venv_dir,
        engine_python,
    }
}

pub fn ensure_setup(bootstrap_python: &str, rt: &ResolvedRuntime) -> Result<()> {
    if rt.engine_python.exists() {
        return Ok(());
    }
    // Create venv + install deps; tools/setup.py also writes active config.
    let status = Command::new(bootstrap_python)
        .args([
            "-m",
            "tools.setup",
            &rt.framework,
            "--venv",
            &rt.venv_dir.to_string_lossy(),
            "--skip-validate",
        ])
        .status()
        .with_context(|| format!("failed to run setup for framework {}", rt.framework))?;
    if !status.success() {
        return Err(anyhow!("setup failed for framework {}", rt.framework));
    }
    Ok(())
}
