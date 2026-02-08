use crate::py_rpc::PyEngine;
use crate::runtime;
use anyhow::{anyhow, Context, Result};
use base64::Engine;
use crossterm::{
    event::{self, Event, KeyCode, KeyEvent, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
    Terminal,
};
use serde_json::json;
use std::{
    io::{self, Stdout},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc,
    },
    thread,
    time::{Duration, Instant},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Screen {
    Landing,
    Explore,
}

#[derive(Debug)]
struct App {
    screen: Screen,
    should_quit: bool,
    bootstrap: String,
    root: std::path::PathBuf,
    rt: runtime::ResolvedRuntime,
    setup_in_flight: bool,
    // Landing/explore shared
    framework: String,
    frameworks: Vec<&'static str>,
    // Transform selection
    all_transforms: Vec<String>,
    cursor: usize,
    selected: Vec<String>, // ordered pipeline
    // Render
    render: RenderState,
    paused: bool,
    last_request_sig: Option<String>,
    last_compute: Instant,
    compute_every: Duration,
}

#[derive(Debug, Clone)]
struct RenderFrame {
    ascii: String,
    meta: String,
}

#[derive(Debug, Default)]
struct RenderState {
    status: String,
    frame: Option<RenderFrame>,
    last_error: Option<String>,
}

#[derive(Debug, Clone)]
struct EngineRequest {
    python: String,
    framework: String,
    transforms: Vec<String>,
}

#[derive(Debug, Clone)]
enum EngineResponse {
    Ok(RenderFrame),
    Err(String),
}

pub fn run_tui(
    bootstrap: &str,
    framework_override: Option<&str>,
    venv_override: Option<&str>,
    transforms: Option<&str>,
) -> Result<()> {
    let root = runtime::repo_root()?;
    let rt = runtime::resolve_runtime(&root, framework_override, venv_override);
    runtime::ensure_setup(&root, bootstrap, &rt)?;

    // List transforms using the active engine python (ensures optional deps can import cleanly later).
    let mut engine_for_list = PyEngine::spawn(rt.engine.to_string_lossy().as_ref())?;
    let transform_keys = {
        let resp = engine_for_list.call("list_transforms", None)?;
        resp.result
            .as_array()
            .ok_or_else(|| anyhow!("unexpected list_transforms response"))?
            .iter()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect::<Vec<_>>()
    };
    engine_for_list.shutdown().ok();

    let (req_tx, req_rx) = mpsc::channel::<EngineRequest>();
    let (resp_tx, resp_rx) = mpsc::channel::<EngineResponse>();
    let (setup_tx, setup_rx) = mpsc::channel::<(String, Result<()>)>();

    // Engine worker thread owns a python process and can swap it when the venv changes.
    let worker = thread::spawn(move || {
        let mut active_python: Option<String> = None;
        let mut engine: Option<PyEngine> = None;
        while let Ok(req) = req_rx.recv() {
            if active_python.as_deref() != Some(&req.python) {
                if let Some(e) = engine.take() {
                    let _ = e.shutdown();
                }
                match PyEngine::spawn(&req.python) {
                    Ok(e) => {
                        active_python = Some(req.python.clone());
                        engine = Some(e);
                    }
                    Err(err) => {
                        let _ = resp_tx.send(EngineResponse::Err(format!("{err:#}")));
                        continue;
                    }
                }
            }

            let Some(engine) = engine.as_mut() else {
                let _ = resp_tx.send(EngineResponse::Err("engine not available".to_string()));
                continue;
            };

            let params = json!({
                "framework": req.framework,
                "transforms": req.transforms.join(","),
                "size": 96,
                "steps": 1,
                "inputs": null,
            });
            let resp = engine.call("run_pipeline", Some(params));
            match resp {
                Ok(r) => match decode_pipeline_result(&r.result) {
                    Ok(frame) => {
                        let _ = resp_tx.send(EngineResponse::Ok(frame));
                    }
                    Err(e) => {
                        let _ = resp_tx.send(EngineResponse::Err(format!("{e:#}")));
                    }
                },
                Err(e) => {
                    let _ = resp_tx.send(EngineResponse::Err(format!("{e:#}")));
                }
            }
        }
        if let Some(e) = engine {
            let _ = e.shutdown();
        }
    });

    // Ctrl+C should return to landing; if already on landing, quit cleanly.
    let got_sigint = Arc::new(AtomicBool::new(false));
    ctrlc::set_handler({
        let got_sigint = got_sigint.clone();
        move || {
            got_sigint.store(true, Ordering::SeqCst);
        }
    })
    .context("failed to install Ctrl+C handler")?;

    let mut terminal = init_terminal()?;
    let mut app = App {
        screen: Screen::Landing,
        should_quit: false,
        bootstrap: bootstrap.to_string(),
        root: root.clone(),
        rt: rt.clone(),
        setup_in_flight: false,
        framework: rt.framework.clone(),
        frameworks: vec!["jax", "numpy", "pytorch", "keras", "cupy", "mlx"],
        all_transforms: transform_keys,
        cursor: 0,
        selected: parse_csv(transforms).unwrap_or_default(),
        render: RenderState {
            status: "Paused. Press 'p' to run.".to_string(),
            frame: None,
            last_error: None,
        },
        paused: true,
        last_request_sig: None,
        last_compute: Instant::now() - Duration::from_secs(3600),
        compute_every: Duration::from_millis(250),
    };

    // Ensure cursor is in range
    if app.cursor >= app.all_transforms.len() {
        app.cursor = 0;
    }

    // Event loop
    let tick_rate = Duration::from_millis(50);
    let mut last_tick = Instant::now();
    while !app.should_quit {
        if got_sigint.swap(false, Ordering::SeqCst) {
            if app.screen == Screen::Landing {
                app.should_quit = true;
            } else {
                app.screen = Screen::Landing;
                app.render.status = "Paused. Press Enter to explore.".to_string();
                app.paused = true;
            }
        }

        // Poll engine responses (non-blocking)
        while let Ok(msg) = resp_rx.try_recv() {
            match msg {
                EngineResponse::Ok(frame) => {
                    app.render.frame = Some(frame);
                    app.render.last_error = None;
                    app.render.status = if app.paused {
                        "Paused. Press 'p' to run.".to_string()
                    } else {
                        "Running...".to_string()
                    };
                }
                EngineResponse::Err(e) => {
                    app.render.last_error = Some(e);
                    app.render.status = "Engine error (see panel)".to_string();
                }
            }
        }

        while let Ok((fw, res)) = setup_rx.try_recv() {
            app.setup_in_flight = false;
            if let Err(e) = res {
                app.render.last_error = Some(format!("setup failed for {fw}: {e:#}"));
                app.render.status = "Setup failed (see panel)".to_string();
            } else {
                app.render.last_error = None;
                app.render.status = format!("Installed {fw}. Press 'p' to run.");
                app.last_compute = Instant::now() - Duration::from_secs(3600);
            }
        }

        terminal.draw(|f| draw_ui(f, &app))?;

        let timeout = tick_rate
            .checked_sub(last_tick.elapsed())
            .unwrap_or(Duration::from_millis(0));
        if event::poll(timeout)? {
            if let Event::Key(key) = event::read()? {
                handle_key(&mut app, key, &req_tx, &setup_tx)?;
            }
        }
        if last_tick.elapsed() >= tick_rate {
            last_tick = Instant::now();
            maybe_compute(&mut app, &req_tx, &setup_tx)?;
        }
    }

    restore_terminal(&mut terminal)?;
    drop(req_tx); // stop worker
    let _ = worker.join();
    Ok(())
}

fn parse_csv(s: Option<&str>) -> Option<Vec<String>> {
    let s = s?.trim();
    if s.is_empty() {
        return None;
    }
    Some(
        s.split(',')
            .map(|x| x.trim())
            .filter(|x| !x.is_empty())
            .map(|x| x.to_string())
            .collect(),
    )
}

fn init_terminal() -> Result<Terminal<CrosstermBackend<Stdout>>> {
    enable_raw_mode().context("enable raw mode")?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen).context("enter alt screen")?;
    let backend = CrosstermBackend::new(stdout);
    Ok(Terminal::new(backend).context("create terminal")?)
}

fn restore_terminal(terminal: &mut Terminal<CrosstermBackend<Stdout>>) -> Result<()> {
    disable_raw_mode().ok();
    execute!(terminal.backend_mut(), LeaveAlternateScreen).ok();
    terminal.show_cursor().ok();
    Ok(())
}

fn handle_key(
    app: &mut App,
    key: KeyEvent,
    req_tx: &mpsc::Sender<EngineRequest>,
    setup_tx: &mpsc::Sender<(String, Result<()>)>,
) -> Result<()> {
    match app.screen {
        Screen::Landing => match key {
            KeyEvent {
                code: KeyCode::Char('q'),
                ..
            } => app.should_quit = true,
            KeyEvent {
                code: KeyCode::Enter,
                ..
            } => {
                app.screen = Screen::Explore;
                app.render.status = "Paused. Press 'p' to run.".to_string();
            }
            KeyEvent {
                code: KeyCode::Left,
                ..
            } => cycle_framework(app, -1),
            KeyEvent {
                code: KeyCode::Right,
                ..
            } => cycle_framework(app, 1),
            _ => {}
        },
        Screen::Explore => match key {
            KeyEvent {
                code: KeyCode::Esc, ..
            } => {
                app.screen = Screen::Landing;
                app.paused = true;
                app.render.status = "Paused. Press Enter to explore.".to_string();
            }
            KeyEvent {
                code: KeyCode::Char('q'),
                ..
            } => app.should_quit = true,
            KeyEvent {
                code: KeyCode::Up,
                modifiers: KeyModifiers::NONE,
                ..
            } => app.cursor = app.cursor.saturating_sub(1),
            KeyEvent {
                code: KeyCode::Down,
                modifiers: KeyModifiers::NONE,
                ..
            } => {
                if !app.all_transforms.is_empty() {
                    app.cursor = (app.cursor + 1).min(app.all_transforms.len() - 1);
                }
            }
            KeyEvent {
                code: KeyCode::Char(' '),
                ..
            } => toggle_current(app),
            KeyEvent {
                code: KeyCode::Char('['),
                ..
            } => move_selected(app, -1),
            KeyEvent {
                code: KeyCode::Char(']'),
                ..
            } => move_selected(app, 1),
            KeyEvent {
                code: KeyCode::Char('p'),
                ..
            } => {
                app.paused = !app.paused;
                app.last_compute = Instant::now() - Duration::from_secs(3600);
                if app.paused {
                    app.render.status = "Paused. Press 'p' to run.".to_string();
                } else {
                    app.render.status = "Running...".to_string();
                    maybe_force_compute(app, req_tx, setup_tx)?;
                }
            }
            KeyEvent {
                code: KeyCode::Left,
                ..
            } => {
                cycle_framework(app, -1);
                app.last_compute = Instant::now() - Duration::from_secs(3600);
                if !app.paused {
                    maybe_force_compute(app, req_tx, setup_tx)?;
                }
            }
            KeyEvent {
                code: KeyCode::Right,
                ..
            } => {
                cycle_framework(app, 1);
                app.last_compute = Instant::now() - Duration::from_secs(3600);
                if !app.paused {
                    maybe_force_compute(app, req_tx, setup_tx)?;
                }
            }
            _ => {}
        },
    }
    Ok(())
}

fn toggle_current(app: &mut App) {
    if app.all_transforms.is_empty() {
        return;
    }
    let key = app.all_transforms[app.cursor].clone();
    if let Some(pos) = app.selected.iter().position(|k| k == &key) {
        app.selected.remove(pos);
    } else {
        app.selected.push(key);
    }
    app.last_request_sig = None;
}

fn move_selected(app: &mut App, delta: isize) {
    if app.all_transforms.is_empty() {
        return;
    }
    let key = app.all_transforms[app.cursor].clone();
    let Some(pos) = app.selected.iter().position(|k| k == &key) else {
        return;
    };
    let new_pos =
        (pos as isize + delta).clamp(0, (app.selected.len().saturating_sub(1)) as isize) as usize;
    if new_pos == pos {
        return;
    }
    let item = app.selected.remove(pos);
    app.selected.insert(new_pos, item);
    app.last_request_sig = None;
}

fn cycle_framework(app: &mut App, dir: isize) {
    let idx = app
        .frameworks
        .iter()
        .position(|f| *f == app.framework)
        .unwrap_or(0);
    let n = app.frameworks.len() as isize;
    let next = (idx as isize + dir).rem_euclid(n) as usize;
    app.framework = app.frameworks[next].to_string();
    app.last_request_sig = None;
    // New framework implies a new runtime/python. Defer setup until we actually compute.
    app.rt = runtime::resolve_runtime(&app.root, Some(&app.framework), None);
}

fn maybe_compute(
    app: &mut App,
    req_tx: &mpsc::Sender<EngineRequest>,
    setup_tx: &mpsc::Sender<(String, Result<()>)>,
) -> Result<()> {
    if app.screen != Screen::Explore {
        return Ok(());
    }
    if app.paused {
        return Ok(());
    }
    if app.selected.is_empty() {
        app.render.status = "Select transforms (Space), then press 'p'.".to_string();
        return Ok(());
    }
    if app.last_compute.elapsed() < app.compute_every {
        return Ok(());
    }
    app.last_compute = Instant::now();
    maybe_force_compute(app, req_tx, setup_tx)
}

fn maybe_force_compute(
    app: &mut App,
    req_tx: &mpsc::Sender<EngineRequest>,
    setup_tx: &mpsc::Sender<(String, Result<()>)>,
) -> Result<()> {
    let sig = format!("{}::{}", app.framework, app.selected.join(","));
    if app.last_request_sig.as_deref() == Some(&sig) && !app.paused {
        // allow periodic refresh even if signature didn't change
    } else {
        app.last_request_sig = Some(sig);
    }
    // Ensure the selected framework environment exists before attempting to compute.
    if !app.rt.engine.exists() && !app.setup_in_flight {
        app.setup_in_flight = true;
        app.render.status = format!("Installing {}...", app.framework);
        let bootstrap = app.bootstrap.clone();
        let rt = app.rt.clone();
        let setup_tx = setup_tx.clone();
        let root = app.root.clone();
        thread::spawn(move || {
            let res = runtime::ensure_setup(&root, &bootstrap, &rt);
            let _ = setup_tx.send((rt.framework.clone(), res));
        });
        return Ok(());
    }

    req_tx.send(EngineRequest {
        python: app.rt.engine.to_string_lossy().to_string(),
        framework: app.framework.clone(),
        transforms: app.selected.clone(),
    })?;
    app.render.status = "Computing...".to_string();
    Ok(())
}

fn draw_ui(f: &mut ratatui::Frame, app: &App) {
    let area = f.area();
    let base = Style::default().fg(Color::Gray);
    let accent = Style::default()
        .fg(Color::Cyan)
        .add_modifier(Modifier::BOLD);

    match app.screen {
        Screen::Landing => {
            let block = Block::default()
                .borders(Borders::ALL)
                .title(Span::styled("tensors-to-consciousness", accent));
            let inner = block.inner(area);
            f.render_widget(block, area);

            let lines = vec![
                Line::from(vec![
                    Span::styled("Framework: ", base),
                    Span::styled(&app.framework, accent),
                ]),
                Line::from(""),
                Line::from(vec![Span::styled("Enter", accent), Span::raw("  Explore")]),
                Line::from(vec![
                    Span::styled("Left/Right", accent),
                    Span::raw("  Switch framework"),
                ]),
                Line::from(vec![Span::styled("Ctrl+C", accent), Span::raw("  Quit")]),
                Line::from(vec![Span::styled("q", accent), Span::raw("  Quit")]),
            ];
            let p = Paragraph::new(lines).wrap(Wrap { trim: true });
            f.render_widget(p, inner);
        }
        Screen::Explore => {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(3),
                    Constraint::Min(0),
                    Constraint::Length(1),
                ])
                .split(area);

            // Header
            let header = Block::default().borders(Borders::ALL).title(Span::styled(
                format!("Explore  [{}/{}]", app.framework, app.selected.len()),
                accent,
            ));
            let header_inner = header.inner(chunks[0]);
            f.render_widget(header, chunks[0]);
            let hdr = Paragraph::new(Line::from(vec![
                Span::styled("Left/Right", accent),
                Span::raw(" framework   "),
                Span::styled("Up/Down", accent),
                Span::raw(" move   "),
                Span::styled("Space", accent),
                Span::raw(" toggle   "),
                Span::styled("[/]", accent),
                Span::raw(" order   "),
                Span::styled("p", accent),
                Span::raw(" run/pause   "),
                Span::styled("Esc", accent),
                Span::raw(" landing   "),
                Span::styled("q", accent),
                Span::raw(" quit"),
            ]));
            f.render_widget(hdr, header_inner);

            // Main: left selector + right render
            let main = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(35), Constraint::Percentage(65)])
                .split(chunks[1]);

            draw_transform_selector(f, main[0], app);
            draw_render_panel(f, main[1], app);

            // Footer status
            let footer = Paragraph::new(Line::from(vec![
                Span::styled("Status: ", base),
                Span::styled(&app.render.status, Style::default().fg(Color::Yellow)),
            ]));
            f.render_widget(footer, chunks[2]);
        }
    }
}

fn draw_transform_selector(f: &mut ratatui::Frame, area: Rect, app: &App) {
    let title = format!("Transforms ({} total)", app.all_transforms.len());
    let block = Block::default().borders(Borders::ALL).title(title);
    let inner = block.inner(area);
    f.render_widget(block, area);

    let mut lines: Vec<Line> = Vec::new();
    if app.all_transforms.is_empty() {
        lines.push(Line::from("No transforms available."));
    } else {
        let max = inner.height as usize;
        let start = app.cursor.saturating_sub(max / 2);
        let end = (start + max).min(app.all_transforms.len());
        for (i, key) in app.all_transforms[start..end].iter().enumerate() {
            let idx = start + i;
            let selected_pos = app.selected.iter().position(|k| k == key);
            let mark = if selected_pos.is_some() { "[x]" } else { "[ ]" };
            let ord = selected_pos
                .map(|p| format!("{:>2}.", p + 1))
                .unwrap_or_else(|| "   ".to_string());
            let style = if idx == app.cursor {
                Style::default()
                    .fg(Color::Black)
                    .bg(Color::Cyan)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::Gray)
            };
            lines.push(Line::from(vec![
                Span::styled(format!("{mark} {ord} "), style),
                Span::styled(key.clone(), style),
            ]));
        }
    }
    let p = Paragraph::new(lines).wrap(Wrap { trim: false });
    f.render_widget(p, inner);
}

fn draw_render_panel(f: &mut ratatui::Frame, area: Rect, app: &App) {
    let block = Block::default().borders(Borders::ALL).title(Span::styled(
        "Shinkei Render",
        Style::default()
            .fg(Color::Magenta)
            .add_modifier(Modifier::BOLD),
    ));
    let inner = block.inner(area);
    f.render_widget(block, area);

    if let Some(err) = &app.render.last_error {
        let p = Paragraph::new(err.clone())
            .style(Style::default().fg(Color::Red))
            .wrap(Wrap { trim: false });
        f.render_widget(p, inner);
        return;
    }
    if let Some(frame) = &app.render.frame {
        let lines = vec![
            Line::from(Span::styled(
                frame.meta.clone(),
                Style::default().fg(Color::DarkGray),
            )),
            Line::from(""),
            Line::from(frame.ascii.clone()),
        ];
        let p = Paragraph::new(lines).wrap(Wrap { trim: false });
        f.render_widget(p, inner);
        return;
    }
    let hint = if app.selected.is_empty() {
        "Select transforms (Space). Press 'p' to run."
    } else if app.paused {
        "Paused. Press 'p' to run."
    } else {
        "Computing..."
    };
    let p = Paragraph::new(hint).style(Style::default().fg(Color::DarkGray));
    f.render_widget(p, inner);
}

fn decode_pipeline_result(v: &serde_json::Value) -> Result<RenderFrame> {
    let shape = v
        .get("shape")
        .and_then(|x| x.as_array())
        .ok_or_else(|| anyhow!("missing shape"))?;
    let h = shape.get(0).and_then(|x| x.as_u64()).context("shape[0]")? as usize;
    let w = shape.get(1).and_then(|x| x.as_u64()).context("shape[1]")? as usize;
    let b64 = v
        .get("data_b64")
        .and_then(|x| x.as_str())
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
    let ascii = crate::ascii_heatmap(&data, h, w, 96, 28);
    let meta = format!("shape={h}x{w} dtype=f32");
    Ok(RenderFrame { ascii, meta })
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
