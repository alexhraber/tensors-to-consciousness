use anyhow::{anyhow, Context, Result};
use base64::Engine;
use clap::{Parser, Subcommand};
use serde_json::json;
use std::io::Write;

mod ops;
mod py_rpc;
mod runtime;
mod shinkei;
mod tui;

#[derive(Parser)]
#[command(
    name = "explorer",
    about = "Rust explorer/TUI host; Python executes transforms/framework math."
)]
struct Cli {
    /// Python interpreter used to bootstrap setup if a framework venv is missing.
    #[arg(long, default_value = "python")]
    bootstrap: String,

    #[command(subcommand)]
    cmd: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// List available transform keys (from Python catalog)
    ListTransforms,

    /// Interactive terminal UI (default)
    Tui {
        /// Framework label (defaults to whatever Python config uses; Rust falls back to "jax")
        #[arg(long)]
        framework: Option<String>,
        /// Venv directory override (default: .venv-<framework>)
        #[arg(long)]
        venv: Option<String>,
        /// Initial transform selection (comma-separated keys)
        #[arg(long)]
        transforms: Option<String>,
    },

    /// Run a transform pipeline once and print a compact ASCII heatmap
    Run {
        #[arg(long)]
        framework: String,
        /// Venv directory override (default: .venv-<framework>)
        #[arg(long)]
        venv: Option<String>,
        #[arg(long)]
        transforms: String,
        #[arg(long, default_value_t = 96)]
        size: usize,
        #[arg(long, default_value_t = 1)]
        steps: usize,
        /// Optional inputs JSON string (same as Python surface)
        #[arg(long)]
        inputs: Option<String>,
    },

    /// Render a tensor payload to a PPM (P6) heatmap (viridis).
    ///
    /// Intended for deterministic asset generation and headless pipelines.
    RenderTensor {
        #[arg(long)]
        h: usize,
        #[arg(long)]
        w: usize,
        /// Base64 of little-endian f32 bytes (row-major), length must be h*w*4.
        #[arg(long)]
        data_b64: String,
        #[arg(long, default_value_t = 320)]
        width_px: u32,
        #[arg(long, default_value_t = 180)]
        height_px: u32,
        /// Output path. If omitted or '-', write PPM bytes to stdout.
        #[arg(long)]
        out: Option<String>,
    },

    /// Validate the selected framework environment (runs Python validation script)
    Validate {
        #[arg(long)]
        framework: Option<String>,
        /// Venv directory override (default: .venv-<framework>)
        #[arg(long)]
        venv: Option<String>,
    },

    /// Repository operations (policy, hooks, PR submission, local CI gate)
    Ops {
        #[command(subcommand)]
        cmd: ops::OpsCmd,
    },
}

pub(crate) fn ascii_heatmap(
    data: &[f32],
    h: usize,
    w: usize,
    out_w: usize,
    out_h: usize,
) -> String {
    let ramp: &[u8] = b" .:-=+*#%@";
    if data.is_empty() || h == 0 || w == 0 {
        return "(empty)".to_string();
    }

    let out_h = out_h.max(1).min(h);
    let out_w = out_w.max(1).min(w);
    let ys = sample_axis(h, out_h);
    let xs = sample_axis(w, out_w);

    let mut mn = f32::INFINITY;
    let mut mx = f32::NEG_INFINITY;
    for &yy in &ys {
        for &xx in &xs {
            let v = data[yy * w + xx];
            if v < mn {
                mn = v;
            }
            if v > mx {
                mx = v;
            }
        }
    }
    let span = mx - mn;

    let mut out = String::with_capacity(out_h * (out_w + 1));
    for (iy, &yy) in ys.iter().enumerate() {
        for &xx in &xs {
            let raw = data[yy * w + xx];
            let norm = if span.abs() < 1e-12 {
                0.5
            } else {
                (raw - mn) / span
            };
            let idx =
                ((norm.clamp(0.0, 1.0) * ((ramp.len() - 1) as f32)) as usize).min(ramp.len() - 1);
            out.push(ramp[idx] as char);
        }
        if iy + 1 < ys.len() {
            out.push('\n');
        }
    }
    out
}

fn sample_axis(in_len: usize, out_len: usize) -> Vec<usize> {
    if in_len == 0 || out_len == 0 {
        return vec![];
    }
    if out_len == 1 {
        return vec![0];
    }
    let mut idx = Vec::with_capacity(out_len);
    let den = (out_len - 1) as f32;
    let top = (in_len - 1) as f32;
    for i in 0..out_len {
        let x = ((i as f32) * top / den).floor() as usize;
        idx.push(x.min(in_len - 1));
    }
    idx
}

fn parse_f32_le_bytes(bytes: &[u8]) -> Result<Vec<f32>> {
    if bytes.len() % 4 != 0 {
        return Err(anyhow!("invalid f32 byte payload (len % 4 != 0)"));
    }
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    // Ensure relative paths (framework scripts, docs, assets) resolve from repo root.
    let root = runtime::repo_root()?;
    std::env::set_current_dir(&root).context("chdir to repo root")?;

    match cli.cmd.unwrap_or(Command::Tui {
        framework: None,
        venv: None,
        transforms: None,
    }) {
        Command::ListTransforms => {
            // Listing transforms does not require a configured venv; it only reads the registry.
            let mut engine = py_rpc::PyEngine::spawn(&cli.bootstrap)?;
            let resp = engine.call("list_transforms", None)?;
            let keys = resp
                .result
                .as_array()
                .ok_or_else(|| anyhow!("unexpected list_transforms response"))?;
            for k in keys {
                if let Some(s) = k.as_str() {
                    println!("{s}");
                }
            }
            engine.shutdown()?;
        }
        Command::Tui {
            framework,
            venv,
            transforms,
        } => {
            tui::run_tui(
                &cli.bootstrap,
                framework.as_deref(),
                venv.as_deref(),
                transforms.as_deref(),
            )?;
        }
        Command::Run {
            framework,
            venv,
            transforms,
            size,
            steps,
            inputs,
        } => {
            let rt = runtime::resolve_runtime(&root, Some(&framework), venv.as_deref());
            runtime::ensure_setup(&cli.bootstrap, &rt)?;
            let mut engine = py_rpc::PyEngine::spawn(rt.engine.to_string_lossy().as_ref())?;
            let params = json!({
                "framework": rt.framework,
                "transforms": transforms,
                "size": size,
                "steps": steps,
                "inputs": inputs,
            });
            let resp = engine.call("run_pipeline", Some(params))?;
            let shape = resp
                .result
                .get("shape")
                .and_then(|v| v.as_array())
                .ok_or_else(|| anyhow!("missing shape"))?;
            let h = shape.get(0).and_then(|v| v.as_u64()).context("shape[0]")? as usize;
            let w = shape.get(1).and_then(|v| v.as_u64()).context("shape[1]")? as usize;
            let b64 = resp
                .result
                .get("data_b64")
                .and_then(|v| v.as_str())
                .ok_or_else(|| anyhow!("missing data_b64"))?;
            let raw = base64::engine::general_purpose::STANDARD
                .decode(b64)
                .context("base64 decode")?;
            let data = parse_f32_le_bytes(&raw)?;
            if data.len() != h * w {
                return Err(anyhow!(
                    "tensor payload size mismatch: got {} expected {}",
                    data.len(),
                    h * w
                ));
            }
            println!("{}", ascii_heatmap(&data, h, w, 96, 28));
            engine.shutdown()?;
        }
        Command::RenderTensor {
            h,
            w,
            data_b64,
            width_px,
            height_px,
            out,
        } => {
            let raw = base64::engine::general_purpose::STANDARD
                .decode(&data_b64)
                .context("base64 decode")?;
            let data = parse_f32_le_bytes(&raw)?;
            if data.len() != h * w {
                return Err(anyhow!(
                    "tensor payload size mismatch: got {} expected {}",
                    data.len(),
                    h * w
                ));
            }
            let ppm = shinkei::render_ppm_viridis(&data, h, w, width_px, height_px)?;
            match out.as_deref() {
                None | Some("-") => {
                    let mut stdout = std::io::stdout().lock();
                    stdout.write_all(&ppm).context("write ppm to stdout")?;
                }
                Some(p) => {
                    std::fs::write(p, &ppm).with_context(|| format!("write ppm to {p}"))?;
                }
            }
        }
        Command::Validate { framework, venv } => {
            let rt = runtime::resolve_runtime(&root, framework.as_deref(), venv.as_deref());
            runtime::ensure_setup(&cli.bootstrap, &rt)?;
            let status = std::process::Command::new(&rt.engine)
                .args([format!("frameworks/{}/test_setup.py", rt.framework)])
                .status()
                .context("run framework test_setup")?;
            if !status.success() {
                return Err(anyhow!("validate failed for framework {}", rt.framework));
            }
        }
        Command::Ops { cmd } => {
            ops::run_ops(cmd)?;
        }
    }
    Ok(())
}
