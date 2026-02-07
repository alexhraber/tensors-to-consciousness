use anyhow::{anyhow, Context, Result};
use clap::{Parser, ValueEnum};
use std::{
    env,
    path::PathBuf,
    process::{Command, Stdio},
};

#[cfg(unix)]
use std::os::unix::process::CommandExt;

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum PullMode {
    /// Pull only when missing (default).
    Missing,
    /// Always attempt `docker pull` before running.
    Always,
    /// Never pull; error if image is missing.
    Never,
}

#[derive(Parser, Debug)]
#[command(
    name = "cargo-explorer",
    about = "Cargo subcommand wrapper: ensure a Docker image and exec `explorer` inside it.",
    after_help = "Usage examples:\n  cargo explorer -- tui\n  cargo explorer -- run --framework jax --transforms default\n  cargo explorer --image ghcr.io/OWNER/REPO/runtime:latest --pull always -- tui\n\nEverything after `--` is passed to the container's `explorer` entrypoint."
)]
struct Cli {
    /// Docker image reference. If omitted, use EXPLORER_IMAGE or a conservative default.
    #[arg(long)]
    image: Option<String>,

    /// Image pull policy.
    #[arg(long, value_enum, default_value_t = PullMode::Missing)]
    pull: PullMode,

    /// Disable mounting the current directory into /workspace.
    #[arg(long)]
    no_volume: bool,

    /// Container workdir (only relevant if volume mount is enabled).
    #[arg(long, default_value = "/workspace")]
    workdir: String,

    /// Extra `docker run` args (repeatable), e.g. `--docker-arg --gpus=all`.
    #[arg(long = "docker-arg")]
    docker_args: Vec<String>,

    /// Arguments passed to `explorer` inside the container.
    #[arg(trailing_var_arg = true)]
    explorer_args: Vec<String>,
}

fn default_image() -> String {
    if let Ok(v) = env::var("EXPLORER_IMAGE") {
        let v = v.trim().to_string();
        if !v.is_empty() {
            return v;
        }
    }
    // Conservative default; callers should override via EXPLORER_IMAGE or --image.
    "explorer:runtime".to_string()
}

fn ensure_docker() -> Result<()> {
    let out = Command::new("docker")
        .args(["version"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .context("docker version")?;
    if !out.success() {
        return Err(anyhow!("docker is required (failed `docker version`)"));
    }
    Ok(())
}

fn docker_image_exists(image: &str) -> bool {
    Command::new("docker")
        .args(["image", "inspect", image])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn docker_pull(image: &str) -> Result<()> {
    let st = Command::new("docker")
        .args(["pull", image])
        .status()
        .with_context(|| format!("docker pull {image}"))?;
    if !st.success() {
        return Err(anyhow!("docker pull failed for {image}"));
    }
    Ok(())
}

fn maybe_pull_image(image: &str, mode: PullMode) -> Result<()> {
    match mode {
        PullMode::Always => docker_pull(image),
        PullMode::Missing => {
            if docker_image_exists(image) {
                Ok(())
            } else {
                docker_pull(image)
            }
        }
        PullMode::Never => {
            if docker_image_exists(image) {
                Ok(())
            } else {
                Err(anyhow!(
                    "docker image not present: {image} (pull disabled via --pull never)"
                ))
            }
        }
    }
}

fn is_tty() -> bool {
    #[cfg(unix)]
    {
        use std::io::IsTerminal;
        std::io::stdin().is_terminal() && std::io::stdout().is_terminal()
    }
    #[cfg(not(unix))]
    {
        false
    }
}

fn cwd() -> Result<PathBuf> {
    env::current_dir().context("current_dir")
}

fn build_docker_run_args(cli: &Cli, image: &str) -> Result<Vec<String>> {
    let mut args: Vec<String> = vec!["run".into(), "--rm".into()];

    if is_tty() {
        args.push("-it".into());
    } else {
        args.push("-i".into());
    }

    if !cli.no_volume {
        let host = cwd()?.to_string_lossy().to_string();
        args.push("-v".into());
        args.push(format!("{host}:{}", cli.workdir));
        args.push("-w".into());
        args.push(cli.workdir.clone());
    }

    for a in &cli.docker_args {
        args.push(a.clone());
    }

    args.push(image.to_string());

    // Container entrypoint is `explorer` (image should provide it).
    args.push("explorer".into());

    // Default to TUI if no args were provided.
    if cli.explorer_args.is_empty() {
        args.push("tui".into());
    } else {
        args.extend(cli.explorer_args.iter().cloned());
    }

    Ok(args)
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    ensure_docker()?;

    let image = cli.image.clone().unwrap_or_else(default_image);
    maybe_pull_image(&image, cli.pull)?;

    let args = build_docker_run_args(&cli, &image)?;

    let mut cmd = Command::new("docker");
    cmd.args(args);

    // Preserve stdio. If this is a TUI, we want docker to own the terminal.
    cmd.stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());

    #[cfg(unix)]
    {
        // Replace this process with docker for correct signal/exit behavior.
        Err(cmd.exec().into())
    }
    #[cfg(not(unix))]
    {
        let st = cmd.status().context("docker run")?;
        std::process::exit(st.code().unwrap_or(1));
    }
}
