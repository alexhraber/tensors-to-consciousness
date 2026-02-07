use numpy::PyReadonlyArrayDyn;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

fn viridis(v: f32) -> (u8, u8, u8) {
    let stops: [(f32, f32, f32); 5] = [
        (68.0, 1.0, 84.0),
        (59.0, 82.0, 139.0),
        (33.0, 145.0, 140.0),
        (94.0, 201.0, 98.0),
        (253.0, 231.0, 37.0),
    ];

    let v = v.clamp(0.0, 1.0);
    let x = v * ((stops.len() - 1) as f32);
    let i = x.floor() as usize;
    if i >= stops.len() - 1 {
        let s = stops[stops.len() - 1];
        return (s.0 as u8, s.1 as u8, s.2 as u8);
    }
    let t = x - (i as f32);
    let a = stops[i];
    let b = stops[i + 1];
    let r = a.0 + t * (b.0 - a.0);
    let g = a.1 + t * (b.1 - a.1);
    let bch = a.2 + t * (b.2 - a.2);
    (r as u8, g as u8, bch as u8)
}

fn flatten_to_2d(arr: PyReadonlyArrayDyn<'_, f32>) -> (Vec<f32>, usize, usize) {
    let view = arr.as_array();
    let shape = view.shape();
    if shape.is_empty() {
        return (Vec::new(), 0, 0);
    }

    let flat: Vec<f32> = view.iter().copied().collect();
    if flat.is_empty() {
        return (Vec::new(), 0, 0);
    }

    match shape.len() {
        1 => (flat, 1, shape[0]),
        2 => (flat, shape[0], shape[1]),
        _ => {
            let h = shape[0];
            let w = flat.len() / h.max(1);
            (flat, h, w)
        }
    }
}

fn sample_axis(in_len: usize, out_len: usize) -> Vec<usize> {
    if in_len == 0 || out_len == 0 {
        return Vec::new();
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

fn interp_row(src: &[f32], in_w: usize, out_w: usize) -> Vec<f32> {
    if out_w == 0 {
        return Vec::new();
    }
    if in_w <= 1 {
        return vec![src.first().copied().unwrap_or(0.0); out_w];
    }

    let mut out = Vec::with_capacity(out_w);
    let den = (out_w - 1) as f32;
    let top = (in_w - 1) as f32;
    for x in 0..out_w {
        let pos = (x as f32) * top / den;
        let l = pos.floor() as usize;
        let r = (l + 1).min(in_w - 1);
        let t = pos - (l as f32);
        let v = src[l] + t * (src[r] - src[l]);
        out.push(v);
    }
    out
}

fn upsample_interp(src: &[f32], in_h: usize, in_w: usize, out_h: usize, out_w: usize) -> Vec<f32> {
    if in_h == 0 || in_w == 0 || out_h == 0 || out_w == 0 {
        return Vec::new();
    }

    let mut tmp = vec![0.0_f32; in_h * out_w];
    for y in 0..in_h {
        let row = &src[y * in_w..(y + 1) * in_w];
        let interp = interp_row(row, in_w, out_w);
        let dst = &mut tmp[y * out_w..(y + 1) * out_w];
        dst.copy_from_slice(&interp);
    }

    if out_h == 1 {
        return tmp[..out_w].to_vec();
    }

    let mut out = vec![0.0_f32; out_h * out_w];
    let den = (out_h - 1) as f32;
    let top = (in_h - 1) as f32;
    for y in 0..out_h {
        let pos = (y as f32) * top / den;
        let t0 = pos.floor() as usize;
        let t1 = (t0 + 1).min(in_h - 1);
        let a = pos - (t0 as f32);
        for x in 0..out_w {
            let v0 = tmp[t0 * out_w + x];
            let v1 = tmp[t1 * out_w + x];
            out[y * out_w + x] = v0 + a * (v1 - v0);
        }
    }
    out
}

#[pyfunction]
fn ascii_heatmap(
    arr: PyReadonlyArrayDyn<'_, f32>,
    width: usize,
    height: usize,
) -> PyResult<String> {
    let ramp: &[u8] = b" .:-=+*#%@";
    let (flat, h, w) = flatten_to_2d(arr);
    if flat.is_empty() || h == 0 || w == 0 {
        return Ok("(empty)".to_string());
    }

    let out_h = h.min(height.max(1));
    let out_w = w.min(width.max(1));
    let ys = sample_axis(h, out_h);
    let xs = sample_axis(w, out_w);

    let mut mn = f32::INFINITY;
    let mut mx = f32::NEG_INFINITY;
    for &yy in &ys {
        for &xx in &xs {
            let v = flat[yy * w + xx];
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
            let raw = flat[yy * w + xx];
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

    Ok(out)
}

#[pyfunction]
fn pixel_heatmap(
    arr: PyReadonlyArrayDyn<'_, f32>,
    width: usize,
    height: usize,
) -> PyResult<String> {
    let (flat, h, w) = flatten_to_2d(arr);
    if flat.is_empty() || h == 0 || w == 0 {
        return Ok("(empty)".to_string());
    }

    let out_w = width.max(2);
    let mut px_h = (height.max(1)) * 2;
    if px_h < 2 {
        px_h = 2;
    }

    let mut sampled = upsample_interp(&flat, h, w, px_h, out_w);
    let mut sampled_h = px_h;
    if sampled_h == 1 {
        let row = sampled.clone();
        sampled.extend_from_slice(&row);
        sampled_h = 2;
    } else if sampled_h % 2 == 1 {
        let start = (sampled_h - 1) * out_w;
        let row = sampled[start..start + out_w].to_vec();
        sampled.extend_from_slice(&row);
        sampled_h += 1;
    }

    let mut mn = f32::INFINITY;
    let mut mx = f32::NEG_INFINITY;
    for &v in &sampled {
        if v < mn {
            mn = v;
        }
        if v > mx {
            mx = v;
        }
    }
    let span = mx - mn;

    let lines = sampled_h / 2;
    let mut out = String::new();
    for row in 0..lines {
        let top_i = (row * 2) * out_w;
        let bot_i = (row * 2 + 1) * out_w;
        for x in 0..out_w {
            let vt = sampled[top_i + x];
            let vb = sampled[bot_i + x];
            let nt = if span.abs() < 1e-12 {
                0.5
            } else {
                (vt - mn) / span
            };
            let nb = if span.abs() < 1e-12 {
                0.5
            } else {
                (vb - mn) / span
            };
            let (tr, tg, tb) = viridis(nt);
            let (br, bg, bb) = viridis(nb);
            out.push_str(&format!(
                "\x1b[38;2;{};{};{}m\x1b[48;2;{};{};{}m▀\x1b[0m",
                tr, tg, tb, br, bg, bb
            ));
        }
        if row + 1 < lines {
            out.push('\n');
        }
    }

    Ok(out)
}

#[pyfunction]
fn parse_assignment(py: Python<'_>, expr: &str) -> PyResult<PyObject> {
    let trimmed = expr.trim();
    let Some((lhs, rhs)) = trimmed.split_once('=') else {
        return Ok(py.None());
    };
    let key = lhs.trim();
    let value = rhs.trim();
    if key.is_empty() || value.is_empty() {
        return Ok(py.None());
    }
    let tuple = PyTuple::new(py, [key.to_string(), value.to_string()])?;
    Ok(tuple.into_any().unbind())
}

#[pyfunction]
fn normalize_platform(raw: Option<&str>, default_platform: &str) -> PyResult<String> {
    let p = raw.unwrap_or(default_platform).trim().to_ascii_lowercase();
    if p == "cpu" || p == "gpu" {
        Ok(p)
    } else {
        Ok(default_platform.to_ascii_lowercase())
    }
}

#[pyfunction]
fn default_venv(framework: &str) -> PyResult<String> {
    Ok(format!(".venv-{}", framework))
}

#[pyfunction]
fn frame_patch(prev: &str, next: &str) -> PyResult<String> {
    if prev.is_empty() {
        return Ok(format!("\x1b[H\x1b[J{}", next));
    }
    if prev == next {
        return Ok(String::new());
    }

    let prev_lines: Vec<&str> = prev.split('\n').collect();
    let next_lines: Vec<&str> = next.split('\n').collect();
    let mut common = 0usize;
    let lim = prev_lines.len().min(next_lines.len());
    while common < lim && prev_lines[common] == next_lines[common] {
        common += 1;
    }

    let mut out = String::from("\x1b[H");
    if common > 0 {
        out.push_str(&format!("\x1b[{}B", common));
    }
    out.push_str("\x1b[J");
    if common < next_lines.len() {
        out.push_str(&next_lines[common..].join("\n"));
    }
    Ok(out)
}

#[pymodule]
fn core(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ascii_heatmap, m)?)?;
    m.add_function(wrap_pyfunction!(pixel_heatmap, m)?)?;
    m.add_function(wrap_pyfunction!(parse_assignment, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_platform, m)?)?;
    m.add_function(wrap_pyfunction!(default_venv, m)?)?;
    m.add_function(wrap_pyfunction!(frame_patch, m)?)?;
    Ok(())
}
