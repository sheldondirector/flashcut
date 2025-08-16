#!/usr/bin/env python3
from __future__ import annotations
from flask import Flask, request, redirect, url_for, render_template_string, send_from_directory, flash
from pathlib import Path
from typing import List, Tuple
from datetime import datetime
import subprocess, shutil, json, os
import numpy as np

# Compatibility shims for deprecated numpy aliases used by some libs
if not hasattr(np, "complex"):
    np.complex = np.complex128  # type: ignore[attr-defined]
if not hasattr(np, "float"):
    np.float = np.float64  # type: ignore[attr-defined]

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FRAME_HOP = 512

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "dev-secret")
BASE_DIR = Path(__file__).parent.resolve()
JOBS_DIR = BASE_DIR / "jobs"
JOBS_DIR.mkdir(exist_ok=True)

INDEX_HTML = """
<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>Flash-cut Builder</title><link rel=\"stylesheet\" href=\"https://cdn.jsdelivr.net/npm/@picocss/pico@2/css/pico.min.css\"><style>body{padding-block:1rem}.grid{display:grid;gap:1rem;grid-template-columns:repeat(auto-fit,minmax(280px,1fr))}.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}.wave{max-width:100%;border-radius:12px;border:1px solid #ddd;background:#fff}</style></head><body><main class=\"container\"><h2>Flash-cut Builder</h2>{% with messages = get_flashed_messages() %}{% if messages %}<article>{% for m in messages %}<p>{{m}}</p>{% endfor %}</article>{% endif %}{% endwith %}<form action=\"{{ url_for('analyze') }}\" method=\"post\" enctype=\"multipart/form-data\"><fieldset><legend>Audio</legend><input type=\"file\" name=\"audio\" accept=\"audio/*\" required></fieldset><div class=\"grid\"><fieldset><legend>Analysis</legend><label>FPS <input type=\"number\" name=\"fps\" step=\"1\" min=\"10\" max=\"120\" value=\"30\"></label><label>Onset threshold <input type=\"number\" name=\"threshold\" step=\"0.01\" min=\"0\" max=\"1\" value=\"0.30\"></label><label>Max gap (s) <input type=\"number\" name=\"max_gap\" step=\"0.05\" min=\"0.1\" max=\"10\" value=\"5.0\"></label></fieldset><fieldset><legend>Flash window</legend><label>Start (s) <input type=\"number\" name=\"flash_start\" step=\"0.1\" value=\"10\"></label><label>End (s) <input type=\"number\" name=\"flash_end\" step=\"0.1\" value=\"25\"></label><label>Min flash gap (s) <input type=\"number\" name=\"flash_gap\" step=\"0.01\" value=\"0.12\"></label></fieldset><fieldset><legend>Render (optional)</legend><label><input type=\"checkbox\" name=\"do_render\" value=\"1\"> Render video with ffmpeg</label><label>Video clips (multiple) <input type=\"file\" name=\"videos\" accept=\"video/*\" multiple></label><label>PNG images (multiple) <input type=\"file\" name=\"images\" accept=\"image/png\" multiple></label><label>Clip portion <select name=\"clip_mode\"><option value=\"head\" selected>Head (start)</option><option value=\"tail\">Tail (end)</option></select></label><label>Output file name <input type=\"text\" name=\"output_name\" value=\"final_video.mp4\"></label></fieldset></div><button type=\"submit\">Analyze</button></form></main></body></html>
"""

RESULT_HTML = """
<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>Flash-cut Result</title><link rel=\"stylesheet\" href=\"https://cdn.jsdelivr.net/npm/@picocss/pico@2/css/pico.min.css\"><style>.grid{display:grid;gap:1rem;grid-template-columns:repeat(auto-fit,minmax(280px,1fr))}.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}.box{border:1px solid #ddd;border-radius:12px;padding:1rem;background:#fff}img.wave{max-width:100%;border-radius:12px;border:1px solid #ddd;background:#fff;max-height:320px;object-fit:cover}</style></head><body><main class=\"container\"><h2>Flash-cut Result</h2><p class=\"mono\">Job: {{ job_id }}</p><div class=\"grid\"><section class=\"box\"><h4>Summary</h4><ul><li>Onsets: <strong>{{ num_onsets }}</strong></li><li>Segments: <strong>{{ num_segments }}</strong></li><li>Flash cuts: <strong>{{ num_flash }}</strong> ({{ flash_start }}–{{ flash_end }} s)</li><li>FPS: {{ fps }}, Threshold: {{ threshold }}, Max gap: {{ max_gap }}</li></ul><div><a href=\"{{ url_for('download', job_id=job_id, filename='cuts.json') }}\">Download cuts.json</a> • <a href=\"{{ url_for('download', job_id=job_id, filename='cuts.csv') }}\">Download cuts.csv</a> • <a href=\"{{ url_for('download', job_id=job_id, filename='waveform.png') }}\">Download waveform.png</a>{% if rendered %} • <a href=\"{{ url_for('download', job_id=job_id, filename=output_name) }}\">Download {{ output_name }}</a>{% endif %}</div></section><section class=\"box\"><h4>Waveform</h4><img class=\"wave\" src=\"{{ url_for('download', job_id=job_id, filename='waveform.png') }}\" alt=\"waveform\"></section></div><details><summary>Segments (first 100)</summary><pre class=\"mono\" style=\"white-space:pre-wrap\">{{ segments_preview }}</pre></details><p><a href=\"{{ url_for('index') }}\">← New analysis</a></p></main></body></html>
"""
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB cap

# ---------------- helpers ----------------

def _ffmpeg_bin() -> str:
    return os.environ.get("FFMPEG") or shutil.which("ffmpeg") or "ffmpeg"

def _ffprobe_bin() -> str:
    return os.environ.get("FFPROBE") or shutil.which("ffprobe") or "ffprobe"

def run_cmd(cmd: List[str], cwd: str | Path | None = None):
    """Run a subprocess and raise with captured logs on failure (friendlier Flask flash)."""
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "FFmpeg failed (exit %s)\nCMD: %s\n--- STDOUT ---\n%s\n--- STDERR ---\n%s" % (
                proc.returncode, " ".join(cmd), proc.stdout or "", proc.stderr or ""
            )
        )
    return proc

# ---------------- signal processing ----------------

def quantize_to_fps(times: List[float], fps: float) -> List[float]:
    return [round(t * fps) / fps for t in times]

def confidence_from_envelope(times, env, sr, hop):
    if len(times) == 0:
        return []
    frames = librosa.time_to_frames(times, sr=sr, hop_length=hop)
    frames = np.clip(frames, 0, len(env) - 1)
    vals = env[frames]
    scale = np.quantile(env, 0.98) or (env.max() or 1.0)
    return np.clip(vals / (scale if scale > 0 else 1.0), 0.0, 1.0).tolist()

def detect_onsets_flux(y: np.ndarray, sr: int, hop: int = FRAME_HOP, threshold: float = 0.30) -> List[dict]:
    _, y_perc = librosa.effects.hpss(y)
    env = librosa.onset.onset_strength(y=y_perc, sr=sr, hop_length=hop, aggregate=np.median)
    onset_times = librosa.onset.onset_detect(
        onset_envelope=env, sr=sr, hop_length=hop, units="time",
        backtrack=False, pre_max=3, post_max=3, pre_avg=3, post_avg=3, delta=0.0
    )
    conf = confidence_from_envelope(onset_times, env, sr, hop)
    keep = [i for i, c in enumerate(conf) if c >= threshold]
    return [{"time": float(onset_times[i]), "confidence": float(conf[i])} for i in keep]

def detect_beats(audio_path: str, threshold: float):
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    duration = float(librosa.get_duration(y=y, sr=sr))
    events = detect_onsets_flux(y, sr, hop=FRAME_HOP, threshold=threshold)
    return events, duration, sr, y

# ---------------- timeline building ----------------

def compute_intervals(beats: List[dict], duration: float, fps: float, max_gap: float):
    end = round(duration * fps) / fps
    if not beats:
        # chunk whole track into <= max_gap spans
        splits = [0.0]
        prev = 0.0
        while end - prev > max_gap:
            prev += max_gap
            splits.append(prev)
        splits.append(end)
        starts = [round(s, 3) for s in splits[:-1]]
        ends = [round(e, 3) for e in splits[1:]]
        return starts, ends
    beat_times = quantize_to_fps(sorted(float(b["time"]) for b in beats), fps)
    splits = [0.0]
    prev = 0.0
    first = beat_times[0]
    while first - prev > max_gap:
        prev += max_gap
        splits.append(prev)
    splits.append(first)
    for i in range(1, len(beat_times)):
        L, R = beat_times[i - 1], beat_times[i]
        prev = L
        while R - prev > max_gap:
            prev += max_gap
            splits.append(prev)
        splits.append(R)
    # ensure tail to full end
    if end > splits[-1]:
        prev = splits[-1]
        while end - prev > max_gap:
            prev += max_gap
            splits.append(prev)
        splits.append(end)
    starts = [round(s, 3) for s in splits[:-1]]
    ends = [round(e, 3) for e in splits[1:]]
    return starts, ends

# ---------------- flash window ----------------

def detect_flash_window(y: np.ndarray, sr: int, window: Tuple[float, float], min_gap: float, fps: float, threshold: float) -> List[float]:
    start_s, end_s = max(0.0, min(window)), max(0.0, max(window))
    i0, i1 = int(start_s * sr), int(end_s * sr)
    seg = y[i0:i1]
    if seg.size == 0:
        return []
    events = detect_onsets_flux(seg, sr, hop=FRAME_HOP, threshold=threshold)
    times = sorted([e["time"] + start_s for e in events])
    pruned, last = [], -1e9
    g = max(1.0 / fps, float(min_gap))
    for t in times:
        if t - last >= g:
            pruned.append(t)
            last = t
    return quantize_to_fps(pruned, fps)

# ---------------- rendering (PNG and VIDEO) ----------------

def render_from_images(pngs: List[str], starts: List[float], ends: List[float], audio: str, fps: float, out_path: str) -> None:
    ffmpeg = _ffmpeg_bin()
    tmp = Path(out_path).parent / "_preconv"
    tmp.mkdir(parents=True, exist_ok=True)
    clip_paths = []
    for i, (s, e) in enumerate(zip(starts, ends), 1):
        length = max(1.0 / fps, e - s)
        src = pngs[(i - 1) % len(pngs)]
        out_i = tmp / f"seg_{i:04d}.mp4"
        cmd = [
            ffmpeg, "-y", "-loop", "1", "-t", f"{length:.3f}", "-i", src,
            "-vf", f"fps={int(fps)},scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2:black",
            "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            str(out_i),
        ]
        run_cmd(cmd)
        clip_paths.append(out_i)
    # concat list with explicit header + LF newlines (Windows friendly)
    list_file = tmp / "list.txt"
    list_text = "ffconcat version 1.0\n" + "\n".join(f"file '{p.name}'" for p in clip_paths) + "\n"
    list_file.write_text(list_text, encoding="utf-8", newline="\n")
    concat_out = tmp / "video.mp4"
    run_cmd([
        ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", "list.txt",
        "-fflags", "+genpts", "-r", str(int(fps)), "-pix_fmt", "yuv420p",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-movflags", "+faststart", str(concat_out),
    ], cwd=tmp)
    run_cmd([ffmpeg, "-y", "-i", str(concat_out), "-i", audio, "-c:v", "copy", "-c:a", "aac", "-shortest", out_path])


def probe_video_meta(path: str):
    ffprobe = _ffprobe_bin()
    try:
        proc = subprocess.run([
            ffprobe, "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,duration", "-show_entries", "format=duration",
            "-of", "json", path,
        ], check=True, capture_output=True, text=True)
        data = json.loads(proc.stdout or "{}")
        w = h = None
        dur = None
        if data.get("streams"):
            s0 = data["streams"][0]
            w = int(s0.get("width") or 0) or None
            h = int(s0.get("height") or 0) or None
            if s0.get("duration"):
                try: dur = float(s0.get("duration"))
                except Exception: pass
        if data.get("format", {}).get("duration"):
            try: dur = float(data["format"]["duration"]) or dur
            except Exception: pass
        if not w or not h: w, h = 1280, 720
        if dur is None: dur = 0.0
        return w, h, float(dur)
    except Exception:
        return 1280, 720, 0.0


def render_from_videos(videos: List[str], starts: List[float], ends: List[float], audio: str, fps: float, out_path: str, clip_mode: str = "head") -> None:
    if not videos:
        raise RuntimeError("No video files provided")
    ffmpeg = _ffmpeg_bin()
    target_w, target_h, _ = probe_video_meta(videos[0])

    tmp = Path(out_path).parent / "_preconv"
    tmp.mkdir(parents=True, exist_ok=True)
    clip_paths: List[Path] = []

    for i, (s, e) in enumerate(zip(starts, ends), 1):
        length = max(1.0 / fps, e - s)
        src = videos[(i - 1) % len(videos)]
        _, _, dur = probe_video_meta(src)
        out_i = tmp / f"seg_{i:04d}.mp4"
        if dur > 0 and length <= dur:
            ss = max(dur - length, 0.0) if clip_mode == "tail" else 0.0
            cmd = [
                ffmpeg, "-y", "-ss", f"{ss:.3f}", "-t", f"{length:.3f}", "-i", src,
                "-vf", f"fps={int(fps)},scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:black",
                "-an", "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                str(out_i),
            ]
        else:
            cmd = [
                ffmpeg, "-y", "-stream_loop", "-1", "-t", f"{length:.3f}", "-i", src,
                "-vf", f"fps={int(fps)},scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:black",
                "-an", "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                str(out_i),
            ]
        run_cmd(cmd)
        clip_paths.append(out_i)

    # Concat (Windows-safe)
    list_file = tmp / "list.txt"
    list_text = "ffconcat version 1.0\n" + "\n".join(f"file '{p.name}'" for p in clip_paths) + "\n"
    list_file.write_text(list_text, encoding="utf-8", newline="\n")
    concat_out = tmp / "video.mp4"
    run_cmd([
        ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", "list.txt",
        "-fflags", "+genpts", "-r", str(int(fps)), "-pix_fmt", "yuv420p",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-movflags", "+faststart", str(concat_out),
    ], cwd=tmp)

    # Mux with audio
    run_cmd([ffmpeg, "-y", "-i", str(concat_out), "-i", audio, "-c:v", "copy", "-c:a", "aac", "-shortest", out_path])

# ---------------- routes ----------------
@app.route("/")
def index():
    return render_template_string(INDEX_HTML)

@app.route("/analyze", methods=["POST"])
def analyze():
    audio_file = request.files.get("audio")
    if not audio_file or audio_file.filename == "":
        flash("Please upload an audio file.")
        return redirect(url_for("index"))

    fps = float(request.form.get("fps", 30))
    threshold = float(request.form.get("threshold", 0.30))
    max_gap = float(request.form.get("max_gap", 5.0))
    flash_start = float(request.form.get("flash_start", 10.0))
    flash_end = float(request.form.get("flash_end", 25.0))
    flash_gap = float(request.form.get("flash_gap", 0.12))
    do_render = request.form.get("do_render") == "1"
    clip_mode = (request.form.get("clip_mode", "head") or "head").lower()
    if clip_mode not in {"head", "tail"}:
        clip_mode = "head"
    output_name = (request.form.get("output_name", "final_video.mp4") or "final_video.mp4").strip()

    job_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    audio_path = job_dir / audio_file.filename
    audio_file.save(str(audio_path))

    events, duration, sr, y = detect_beats(str(audio_path), threshold=threshold)
    starts, ends = compute_intervals(events, duration, fps, max_gap)

    flash_times = detect_flash_window(y, sr, (flash_start, flash_end), flash_gap, fps, threshold) if flash_end > flash_start else []
    if flash_times:
        # If we later remove flash window UI, leave this here behind a feature flag
        starts, ends = inject_flash_splits(starts, ends, flash_times, fps)

    data = {
        "audio": audio_file.filename,
        "fps": fps,
        "max_gap": max_gap,
        "events_onsets": events,
        "segments": [{"start": s, "end": e} for s, e in zip(starts, ends)],
        "flash": flash_times,
        "flash_window": [flash_start, flash_end],
        "clip_mode": clip_mode,
    }
    (job_dir / "cuts.json").write_text(json.dumps(data, indent=2), encoding="utf-8")

    (job_dir / "cuts.csv").write_text(
        "index,start,end\n" + "\n".join(f"{i+1},{s:.3f},{e:.3f}" for i, (s, e) in enumerate(zip(starts, ends))),
        encoding="utf-8",
    )

    plot_waveform(job_dir / "waveform.png", y, sr, flash_times, (flash_start, flash_end))

    rendered = False
    if do_render:
        saved_videos: List[str] = []
        for f in request.files.getlist("videos"):
            if f and f.filename:
                dst = job_dir / f.filename
                f.save(str(dst))
                saved_videos.append(str(dst))

        if saved_videos:
            out_path = job_dir / output_name
            try:
                render_from_videos(saved_videos, starts, ends, str(audio_path), fps, str(out_path), clip_mode=clip_mode)
                rendered = True
            except Exception as e:
                flash(f"ffmpeg video render failed:\n{e}")
        else:
            # PNG fallback
            images = request.files.getlist("images")
            pngs: List[str] = []
            for f in images:
                if f and f.filename.lower().endswith(".png"):
                    dst = job_dir / f.filename
                    f.save(str(dst))
                    pngs.append(str(dst))
            if pngs:
                out_path = job_dir / output_name
                try:
                    render_from_images(pngs, starts, ends, str(audio_path), fps, str(out_path))
                    rendered = True
                except Exception as e:
                    flash(f"ffmpeg image render failed:\n{e}")
            else:
                flash("No videos or PNGs were provided for rendering.")

    preview_lines = []
    for i, (s, e) in enumerate(zip(starts, ends)):
        if i >= 100:
            preview_lines.append("... (truncated)")
            break
        preview_lines.append(f"{i+1:03d}: {s:.3f} – {e:.3f}")
    segments_preview = "\n".join(preview_lines)

    return render_template_string(
        RESULT_HTML,
        job_id=job_id,
        fps=fps, threshold=threshold, max_gap=max_gap,
        flash_start=flash_start, flash_end=flash_end,
        num_onsets=len(events), num_segments=len(starts), num_flash=len(flash_times),
        segments_preview=segments_preview,
        rendered=rendered, output_name=output_name
    )

@app.route("/jobs/<job_id>/<path:filename>")
def download(job_id, filename):
    folder = JOBS_DIR / job_id
    if not folder.exists():
        return "Not found", 404
    return send_from_directory(folder, filename, as_attachment=True)

def inject_flash_splits(starts: List[float], ends: List[float], flash_times: List[float], fps: float):
    if not flash_times:
        return starts, ends
    flash = sorted(quantize_to_fps(flash_times, fps))
    out_s, out_e = [], []
    for s, e in zip(starts, ends):
        cuts = [t for t in flash if s < t < e]
        if not cuts:
            out_s.append(s)
            out_e.append(e)
            continue
        pts = [s] + cuts + [e]
        for i in range(len(pts) - 1):
            a, b = pts[i], pts[i + 1]
            if b - a < 1.0 / fps:
                b = a + 1.0 / fps
            out_s.append(round(a, 3))
            out_e.append(round(b, 3))
    return out_s, out_e


def plot_waveform(png_path: Path, y: np.ndarray, sr: int, flash_times: List[float], window: Tuple[float, float]):
    t = np.linspace(0, librosa.get_duration(y=y, sr=sr), num=len(y), endpoint=True)
    plt.figure(figsize=(18, 4))
    plt.fill_between(t, y, -y, color="#f0b429", alpha=0.25)
    plt.plot(t, y, color="#f0b429", lw=0.7, alpha=0.8)
    lo, hi = min(window), max(window)
    plt.axvline(lo, color="red", ls="--", lw=2, dashes=(6, 6))
    plt.axvline(hi, color="red", ls="--", lw=2, dashes=(6, 6))
    for x in flash_times:
        plt.axvline(x, color="#22c55e", ls=(0, (3, 5)), lw=1.4, alpha=0.9)
    plt.title("Waveform with Flash Cut Points")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.25, ls="--")
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()
# --- keep everything above as-is ---

@app.route("/health")
def health():
    return {"ok": True}

if __name__ == "__main__":
    import os
    import matplotlib
    matplotlib.use("Agg")  # headless on Railway
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)

