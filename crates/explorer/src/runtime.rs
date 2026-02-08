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

pub fn repo_root_strict() -> Result<PathBuf> {
    if let Ok(v) = std::env::var("EXPLORER_ROOT") {
        let v = v.trim();
        if !v.is_empty() {
            return Ok(PathBuf::from(v));
        }
    }
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

pub fn repo_root() -> Result<PathBuf> {
    // Explorer can run outside a git checkout (Docker bind mounts, packaged binaries).
    // Prefer a strict git-derived root when available, but fall back to cwd.
    if let Ok(p) = repo_root_strict() {
        return Ok(p);
    }
    Ok(std::env::current_dir().context("current_dir")?)
}

pub fn load_active_config(root: &Path) -> Option<ActiveConfig> {
    let new_p = root.join(".explorer").join("config.json");
    if let Ok(raw) = fs::read_to_string(&new_p) {
        if let Ok(cfg) = serde_json::from_str::<ActiveConfig>(&raw) {
            return Some(cfg);
        }
    }
    None
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
    pub engine: PathBuf,
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

    let engine = venv_python(&venv_dir);
    ResolvedRuntime {
        framework,
        venv_dir,
        engine,
    }
}

pub fn ensure_setup(root: &Path, bootstrap: &str, rt: &ResolvedRuntime) -> Result<()> {
    if rt.engine.exists() {
        return Ok(());
    }
    if Command::new("uv").args(["--version"]).output().is_err() {
        return Err(anyhow!(
            "`uv` is required to set up framework environments (install uv, then retry)"
        ));
    }

    // Create venv and install deps.
    let status = Command::new("uv")
        .args(["venv", &rt.venv_dir.to_string_lossy()])
        .status()
        .with_context(|| format!("failed to create venv {}", rt.venv_dir.display()))?;
    if !status.success() {
        return Err(anyhow!("uv venv failed for {}", rt.venv_dir.display()));
    }

    let common_deps: &[&str] = &["matplotlib"];
    let mut framework_deps: Vec<String> = match rt.framework.as_str() {
        // On Linux containers (including Apple Docker Desktop VMs), install MLX CPU backend.
        "mlx" => {
            if cfg!(target_os = "macos") {
                vec!["mlx".to_string()]
            } else {
                vec!["mlx[cpu]".to_string()]
            }
        }
        "jax" => vec!["jax[cpu]".to_string()],
        "pytorch" => vec!["torch".to_string()],
        "numpy" => vec!["numpy".to_string()],
        "keras" => vec!["keras".to_string(), "tensorflow".to_string()],
        "cupy" => vec!["cupy-cuda12x".to_string()],
        other => return Err(anyhow!("unsupported framework: {other}")),
    };

    let mut args: Vec<String> = vec![
        "pip".to_string(),
        "install".to_string(),
        "--python".to_string(),
        rt.engine.to_string_lossy().to_string(),
        "--upgrade".to_string(),
    ];
    args.extend(common_deps.iter().map(|s| s.to_string()));
    args.append(&mut framework_deps);

    let status = Command::new("uv")
        .args(args)
        .status()
        .with_context(|| format!("failed to install deps for framework {}", rt.framework))?;
    if !status.success() {
        return Err(anyhow!(
            "dependency install failed for framework {}",
            rt.framework
        ));
    }

    // Record active config.
    save_active_config(
        root,
        &ActiveConfig {
            framework: rt.framework.clone(),
            venv: rt.venv_dir.to_string_lossy().to_string(),
        },
    )?;

    // Optional validation (best-effort): we keep it cheap for auto-setup.
    let _ = Command::new(&rt.engine)
        .arg(format!("frameworks/{}/test_setup.py", rt.framework))
        .status();

    let _ = bootstrap; // reserved for future: alternate setup strategies
    Ok(())
}

pub fn save_active_config(root: &Path, cfg: &ActiveConfig) -> Result<()> {
    let dir = root.join(".explorer");
    fs::create_dir_all(&dir).context("create .explorer")?;
    let p = dir.join("config.json");
    fs::write(&p, serde_json::to_string_pretty(cfg)?).context("write config.json")?;
    Ok(())
}
