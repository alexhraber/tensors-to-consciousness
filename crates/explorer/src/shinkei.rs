use anyhow::{anyhow, Result};

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

fn percentile_abs_approx(data: &[f32], p: f32) -> f32 {
    if data.is_empty() {
        return 1.0;
    }
    let mut vals: Vec<f32> = data
        .iter()
        .map(|v| if v.is_finite() { v.abs() } else { 0.0 })
        .collect();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p = p.clamp(0.0, 100.0) / 100.0;
    let idx = ((p * ((vals.len() - 1) as f32)).round() as usize).min(vals.len() - 1);
    let out = vals[idx];
    if out.is_finite() && out > 1e-6 { out } else { 1.0 }
}

fn smooth_3x3_edge(src: &[f32], h: usize, w: usize) -> Vec<f32> {
    if h == 0 || w == 0 || src.is_empty() {
        return vec![];
    }
    let mut out = vec![0.0f32; h * w];
    let idx = |yy: isize, xx: isize| -> usize {
        let y = yy.clamp(0, (h as isize) - 1) as usize;
        let x = xx.clamp(0, (w as isize) - 1) as usize;
        y * w + x
    };
    for y in 0..h as isize {
        for x in 0..w as isize {
            let c = src[idx(y, x)];
            let sum = src[idx(y - 1, x - 1)]
                + src[idx(y - 1, x)]
                + src[idx(y - 1, x + 1)]
                + src[idx(y, x - 1)]
                + (2.0 * c)
                + src[idx(y, x + 1)]
                + src[idx(y + 1, x - 1)]
                + src[idx(y + 1, x)]
                + src[idx(y + 1, x + 1)];
            out[(y as usize) * w + (x as usize)] = sum / 10.0;
        }
    }
    out
}

fn upsample_bilinear(src: &[f32], in_h: usize, in_w: usize, out_h: usize, out_w: usize) -> Vec<f32> {
    if in_h == 0 || in_w == 0 || out_h == 0 || out_w == 0 {
        return vec![];
    }
    if in_h == out_h && in_w == out_w {
        return src.to_vec();
    }
    let mut out = vec![0.0f32; out_h * out_w];
    let h_scale = if out_h <= 1 { 0.0 } else { (in_h - 1) as f32 / (out_h - 1) as f32 };
    let w_scale = if out_w <= 1 { 0.0 } else { (in_w - 1) as f32 / (out_w - 1) as f32 };
    for y in 0..out_h {
        let fy = (y as f32) * h_scale;
        let y0 = fy.floor() as usize;
        let y1 = (y0 + 1).min(in_h - 1);
        let ty = fy - (y0 as f32);
        for x in 0..out_w {
            let fx = (x as f32) * w_scale;
            let x0 = fx.floor() as usize;
            let x1 = (x0 + 1).min(in_w - 1);
            let tx = fx - (x0 as f32);
            let a = src[y0 * in_w + x0];
            let b = src[y0 * in_w + x1];
            let c = src[y1 * in_w + x0];
            let d = src[y1 * in_w + x1];
            let top = a + tx * (b - a);
            let bot = c + tx * (d - c);
            out[y * out_w + x] = top + ty * (bot - top);
        }
    }
    out
}

pub fn render_ppm_viridis(
    data: &[f32],
    h: usize,
    w: usize,
    width_px: u32,
    height_px: u32,
) -> Result<Vec<u8>> {
    if data.is_empty() || h == 0 || w == 0 {
        return Err(anyhow!("empty tensor"));
    }
    let width_px = width_px.max(2) as usize;
    let height_px = height_px.max(2) as usize;

    // Sanitize.
    let mut buf: Vec<f32> = data
        .iter()
        .map(|v| if v.is_finite() { v.clamp(-1.0e4, 1.0e4) } else { 0.0 })
        .collect();

    // Tone-map for calmer previews, then smooth (same intent as legacy Python Shinkei path).
    let scale = percentile_abs_approx(&buf, 92.0);
    for v in &mut buf {
        *v = ((*v) / scale).tanh() * 2.0;
    }
    let sm = smooth_3x3_edge(&buf, h, w);
    let sampled = upsample_bilinear(&sm, h, w, height_px, width_px);

    // Normalize.
    let mut mn = f32::INFINITY;
    let mut mx = f32::NEG_INFINITY;
    for &v in &sampled {
        let v = if v.is_finite() { v } else { 0.0 };
        if v < mn {
            mn = v;
        }
        if v > mx {
            mx = v;
        }
    }
    let span = (mx - mn).abs().max(1e-12);

    // PPM (P6) header + raw RGB.
    let mut out = Vec::with_capacity(32 + width_px * height_px * 3);
    out.extend_from_slice(format!("P6\n{} {}\n255\n", width_px, height_px).as_bytes());
    for y in 0..height_px {
        for x in 0..width_px {
            let v = sampled[y * width_px + x];
            let norm = ((v - mn) / span).clamp(0.0, 1.0);
            let (r, g, b) = viridis(norm);
            out.push(r);
            out.push(g);
            out.push(b);
        }
    }
    Ok(out)
}

