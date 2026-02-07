from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import tempfile
import time
from pathlib import Path


class HeadlessCaptureError(RuntimeError):
    pass


def _require_binary(name: str) -> None:
    if shutil.which(name) is None:
        raise HeadlessCaptureError(f"{name} is required for headless capture but was not found in PATH.")


def _run(cmd: list[str], *, env: dict[str, str] | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, env=env, check=check, text=True, capture_output=True)


def _window_id(display: str, title: str, *, timeout_s: float = 8.0) -> str:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        result = _run(["xdotool", "search", "--name", title], env={"DISPLAY": display}, check=False)
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if lines:
            return lines[0]
        time.sleep(0.1)
    raise HeadlessCaptureError(f"Timed out waiting for xterm window '{title}'.")


def _send_key(display: str, window_id: str, key: str) -> None:
    _run(["xdotool", "key", "--window", window_id, key], env={"DISPLAY": display})


def _encode_mp4_to_gif(mp4: Path, output_gif: Path, *, fps: int) -> None:
    palette = mp4.with_name("palette.png")
    _run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(mp4),
            "-vf",
            f"fps={fps},scale=960:-1:flags=lanczos,palettegen",
            str(palette),
        ]
    )
    _run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(mp4),
            "-i",
            str(palette),
            "-lavfi",
            f"fps={fps},scale=960:-1:flags=lanczos[x];[x][1:v]paletteuse",
            str(output_gif),
        ]
    )


def capture_command_gif(
    *,
    output_gif: Path,
    command: str,
    width: int = 1280,
    height: int = 720,
    fps: int = 18,
    duration_s: float = 8.0,
    title: str = "explorer-headless-capture",
    key_script: list[tuple[float, str]] | None = None,
    hold: bool = True,
) -> None:
    for dep in ("Xvfb", "xterm", "xdotool", "ffmpeg"):
        _require_binary(dep)

    output_gif.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="explorer_headless_capture_") as td:
        tmp = Path(td)
        mp4 = tmp / "capture.mp4"

        display_num = 90 + (os.getpid() % 100)
        display = f":{display_num}"
        env_display = {"DISPLAY": display}

        xvfb = subprocess.Popen(
            ["Xvfb", display, "-screen", "0", f"{width}x{height}x24", "-nolisten", "tcp"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        xterm = None
        ffmpeg = None
        try:
            time.sleep(0.4)
            xterm_cmd = [
                "xterm",
                "-title",
                title,
                "-geometry",
                "180x52+0+0",
                "-fa",
                "Monospace",
                "-fs",
                "12",
            ]
            if hold:
                xterm_cmd.append("-hold")
            xterm_cmd.extend(["-e", "bash", "-lc", command])

            xterm = subprocess.Popen(
                xterm_cmd,
                env={**os.environ, **env_display},
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            ffmpeg = subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "x11grab",
                    "-video_size",
                    f"{width}x{height}",
                    "-framerate",
                    str(fps),
                    "-i",
                    display,
                    "-t",
                    str(duration_s),
                    "-pix_fmt",
                    "yuv420p",
                    str(mp4),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            win = _window_id(display, title)
            for delay_s, key in (key_script or []):
                time.sleep(delay_s)
                _send_key(display, win, key)

            if ffmpeg.poll() is None:
                ffmpeg.wait(timeout=max(1.0, duration_s + 2.0))

            if not mp4.exists() or mp4.stat().st_size == 0:
                raise HeadlessCaptureError("Headless capture produced no video output.")

            _encode_mp4_to_gif(mp4, output_gif, fps=fps)
        finally:
            for proc in (xterm, ffmpeg, xvfb):
                if proc is None:
                    continue
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=2)

    if not output_gif.exists() or output_gif.stat().st_size == 0:
        raise HeadlessCaptureError("Headless capture did not produce GIF output.")


def capture_tui_session_gif(
    *,
    output_gif: Path,
    python_exe: str,
    framework: str = "jax",
    transforms: str = "default",
    width: int = 1280,
    height: int = 720,
    fps: int = 18,
    duration_s: float = 8.0,
) -> None:
    cmd = (
        "export TERM=xterm-256color COLORTERM=truecolor PYTHONUNBUFFERED=1; "
        f"{shlex.quote(python_exe)} -m tools.tui --framework {shlex.quote(framework)} --transforms {shlex.quote(transforms)}"
    )
    capture_command_gif(
        output_gif=output_gif,
        command=cmd,
        width=width,
        height=height,
        fps=fps,
        duration_s=duration_s,
        title="explorer-tui-capture",
        key_script=[
            (0.8, "Return"),
            (0.8, "n"),
            (0.5, "x"),
            (0.5, "bracketright"),
            (0.5, "space"),
            (1.2, "h"),
            (0.9, "space"),
            (0.8, "q"),
        ],
    )


def capture_transform_progression_gif(
    *,
    output_gif: Path,
    python_exe: str,
    framework: str,
    transforms: str,
    inputs: str | None = None,
    width: int = 1280,
    height: int = 720,
    fps: int = 16,
    duration_s: float = 7.5,
    title: str = "explorer-progression-capture",
) -> None:
    cmd_parts = [
        "export TERM=xterm-256color COLORTERM=truecolor PYTHONUNBUFFERED=1;",
        shlex.quote(python_exe),
        "explorer.py",
        "run",
        "--framework",
        shlex.quote(framework),
        "--transforms",
        shlex.quote(transforms),
    ]
    if inputs:
        cmd_parts.extend(["--inputs", shlex.quote(inputs)])
    cmd = " ".join(cmd_parts)

    capture_command_gif(
        output_gif=output_gif,
        command=cmd,
        width=width,
        height=height,
        fps=fps,
        duration_s=duration_s,
        title=title,
        key_script=[],
        hold=True,
    )
