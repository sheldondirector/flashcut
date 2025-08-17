#!/usr/bin/env python3
from __future__ import annotations

# ---------------- stdlib ----------------
from flask import Flask, request, redirect, url_for, render_template_string, send_from_directory, flash
from pathlib import Path
from typing import List, Tuple
from datetime import datetime
import subprocess, shutil, json, os, sys, platform, textwrap, re, time

# ---------------- numeric / plotting ----------------
import numpy as np

# Compatibility shims for deprecated numpy aliases used by some libs
if not hasattr(np, "complex"):
    np.complex = np.complex128  # type: ignore[attr-defined]
if not hasattr(np, "float"):
    np.float = np.float64  # type: ignore[attr-defined]

# Heavy deps we want to verify in diagnostics:
diag_errors = {}
try:
    import librosa
except Exception as e:
    librosa = None  # type: ignore
    diag_errors["librosa_import_error"] = repr(e)

try:
    import soundfile as sf
except Exception as e:
    sf = None  # type: ignore
    diag_errors["soundfile_import_error"] = repr(e)

try:
    import audioread
except Exception as e:
    audioread = None  # type: ignore
    diag_errors["audioread_import_error"] = repr(e)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as e:
    matplotlib = None  # type: ignore
    plt = None  # type: ignore
    diag_errors["matplotlib_import_error"] = repr(e)

FRAME_HOP = 512

# Mobile optimization settings
MOBILE_MAX_DURATION = 60.0  # Max 60 seconds for mobile
MOBILE_SAMPLE_RATE = 22050  # Reduced sample rate for mobile
MOBILE_FRAME_HOP = 1024     # Larger hop for mobile (faster processing)
MOBILE_PLOT_DPI = 100       # Lower DPI for mobile plots
DESKTOP_PLOT_DPI = 150      # Higher DPI for desktop

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "dev-secret")
BASE_DIR = Path(__file__).parent.resolve()
JOBS_DIR = BASE_DIR / "jobs"
JOBS_DIR.mkdir(exist_ok=True)

# ---------------- diagnostics helpers ----------------

def is_mobile_device() -> bool:
    """Detect if the request comes from a mobile device"""
    user_agent = request.headers.get('User-Agent', '').lower()
    mobile_patterns = [
        r'mobile', r'android', r'iphone', r'ipad', r'ipod', 
        r'blackberry', r'nokia', r'opera mini', r'palm', 
        r'windows phone', r'kindle', r'silk', r'fennec'
    ]
    return any(re.search(pattern, user_agent) for pattern in mobile_patterns)

def get_processing_config() -> dict:
    """Get processing configuration based on device type"""
    if is_mobile_device():
        return {
            'max_duration': MOBILE_MAX_DURATION,
            'sample_rate': MOBILE_SAMPLE_RATE,
            'hop_length': MOBILE_FRAME_HOP,
            'plot_dpi': MOBILE_PLOT_DPI,
            'device_type': 'mobile'
        }
    else:
        return {
            'max_duration': float('inf'),
            'sample_rate': None,  # Let librosa choose
            'hop_length': FRAME_HOP,
            'plot_dpi': DESKTOP_PLOT_DPI,
            'device_type': 'desktop'
        }

def log_performance(operation: str, duration: float, device_type: str = None):
    """Log performance metrics for debugging mobile issues"""
    if device_type is None:
        device_type = 'mobile' if is_mobile_device() else 'desktop'
    
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'operation': operation,
        'duration_seconds': round(duration, 3),
        'device_type': device_type,
        'user_agent': request.headers.get('User-Agent', 'unknown')[:200]
    }
    
    # Log to a simple text file for Railway deployment
    log_dir = BASE_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "performance.log"
    
    try:
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    except Exception:
        pass  # Don't fail if logging fails

def _ffmpeg_bin() -> str:
    return os.environ.get("FFMPEG") or shutil.which("ffmpeg") or "ffmpeg"

def _ffprobe_bin() -> str:
    return os.environ.get("FFPROBE") or shutil.which("ffprobe") or "ffprobe"

def _cmd_ok(bin_name: str, *args: str, timeout: float = 3.0) -> Tuple[bool, str]:
    try:
        proc = subprocess.run([bin_name, *args], text=True, capture_output=True, timeout=timeout)
        ok = proc.returncode == 0
        out = (proc.stdout or "")[:4000]
        err = (proc.stderr or "")[:4000]
        return ok, (out or err) or ""
    except Exception as e:
        return False, repr(e)

def _soundfile_ok() -> Tuple[bool, str]:
    # Check libsndfile presence by asking available formats.
    try:
        if sf is None:
            return False, "soundfile not importable"
        fmts = sf.available_formats()
        ok = bool(fmts)
        info = "formats=" + ",".join(sorted(list(fmts.keys()))[:12])
        # Also probe default library name if possible
        lib = getattr(sf, "_libname", None)
        if lib:
            info += f" lib={lib}"
        return ok, info
    except Exception as e:
        return False, repr(e)

def get_diag() -> dict:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    ff_ok, ff_ver = _cmd_ok(_ffmpeg_bin(), "-version")
    fp_ok, fp_ver = _cmd_ok(_ffprobe_bin(), "-version")

    sf_ok, sf_info = _soundfile_ok()

    # quick numpy/BLAS info (non-fatal)
    try:
        np_show_config = ""
        try:
            from io import StringIO
            buf = StringIO()
            np.show_config(print_to=buf)  # type: ignore[arg-type]
            np_show_config = buf.getvalue()
        except Exception:
            pass
    except Exception:
        np_show_config = ""

    # Mobile-specific diagnostics
    mobile_detected = is_mobile_device()
    config = get_processing_config()
    user_agent = request.headers.get('User-Agent', 'unknown')

    diag = {
        "python": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.release()}",
        "mobile": {
            "is_mobile": mobile_detected,
            "device_type": config['device_type'],
            "user_agent": user_agent[:200],
            "max_duration": config['max_duration'],
            "sample_rate": config['sample_rate'],
            "hop_length": config['hop_length'],
            "plot_dpi": config['plot_dpi'],
        },
        "env": {
            "PYTHON_VERSION": os.environ.get("PYTHON_VERSION"),
            "NIXPACKS_PKGS": os.environ.get("NIXPACKS_PKGS"),
            "PATH_has_ffmpeg": bool(ffmpeg),
            "PATH_has_ffprobe": bool(ffprobe),
        },
        "versions": {
            "numpy": getattr(np, "__version__", None),
            "librosa": getattr(librosa, "__version__", None) if librosa else None,
            "soundfile": getattr(sf, "__version__", None) if sf else None,
            "audioread": getattr(audioread, "__version__", None) if audioread else None,
            "matplotlib": getattr(matplotlib, "__version__", None) if matplotlib else None,
        },
        "bins": {
            "ffmpeg_found": bool(ffmpeg),
            "ffprobe_found": bool(ffprobe),
            "ffmpeg_works": ff_ok,
            "ffprobe_works": fp_ok,
            "ffmpeg_version_head": (ff_ver or "")[:200],
            "ffprobe_version_head": (fp_ver or "")[:200],
        },
        "audio_backends": {
            "libsndfile_ok": sf_ok,
            "libsndfile_info": sf_info,
            "audioread_available": audioread is not None,
        },
        "numpy_config_snippet": (textwrap.shorten(np_show_config.replace("\n", " | "), width=800) if np_show_config else None),
        "import_errors": diag_errors or None,
        "recommendations": [
            "On Railway, set env var NIXPACKS_PKGS='ffmpeg libsndfile' so MP3/M4A/WAV decode works.",
            "Pin wheels in requirements.txt (numpy==1.26.4, librosa==0.10.2.post1, soundfile==0.12.1, audioread==3.0.1, matplotlib==3.8.4).",
            "Gunicorn: --timeout 600 --threads 4 for long analyses.",
        ] + ([
            f"Mobile device detected: Audio limited to {config['max_duration']}s, using optimized settings for better performance.",
            "For longer files or full features, consider using a desktop browser.",
        ] if mobile_detected else []),
    }
    return diag

def render_diag_html(diag: dict) -> str:
    # Pretty HTML card for inline display
    def b(v: bool) -> str:
        return f'<span style="color:{("#16a34a" if v else "#dc2626")};font-weight:600;">{"OK" if v else "MISSING"}</span>'
    v = diag.get("versions", {})
    bins = diag.get("bins", {})
    ab = diag.get("audio_backends", {})
    env = diag.get("env", {})
    mobile = diag.get("mobile", {})
    imports = diag.get("import_errors") or {}

    rows = []
    rows.append(f"<tr><td>Python</td><td>{diag.get('python')}</td></tr>")
    rows.append(f"<tr><td>Platform</td><td>{diag.get('platform')}</td></tr>")
    
    # Mobile-specific diagnostics
    device_color = "#16a34a" if mobile.get('device_type') == 'desktop' else "#f59e0b"
    rows.append(f"<tr><td>Device Type</td><td><span style=\"color:{device_color};font-weight:600;\">{mobile.get('device_type', 'unknown').upper()}</span></td></tr>")
    if mobile.get('is_mobile'):
        rows.append(f"<tr><td>Max Duration</td><td>{mobile.get('max_duration')}s</td></tr>")
        rows.append(f"<tr><td>Sample Rate</td><td>{mobile.get('sample_rate')}Hz</td></tr>")
        rows.append(f"<tr><td>Hop Length</td><td>{mobile.get('hop_length')}</td></tr>")
        rows.append(f"<tr><td>Plot DPI</td><td>{mobile.get('plot_dpi')}</td></tr>")
    
    rows.append(f"<tr><td>numpy</td><td>{v.get('numpy')}</td></tr>")
    rows.append(f"<tr><td>librosa</td><td>{v.get('librosa')}</td></tr>")
    rows.append(f"<tr><td>soundfile</td><td>{v.get('soundfile')}</td></tr>")
    rows.append(f"<tr><td>audioread</td><td>{v.get('audioread')}</td></tr>")
    rows.append(f"<tr><td>matplotlib</td><td>{v.get('matplotlib')}</td></tr>")
    rows.append(f"<tr><td>ffmpeg</td><td>{b(bins.get('ffmpeg_works'))} <small>{bins.get('ffmpeg_version_head','')}</small></td></tr>")
    rows.append(f"<tr><td>ffprobe</td><td>{b(bins.get('ffprobe_works'))} <small>{bins.get('ffprobe_version_head','')}</small></td></tr>")
    rows.append(f"<tr><td>libsndfile</td><td>{b(ab.get('libsndfile_ok'))} <small>{ab.get('libsndfile_info','')}</small></td></tr>")
    rows.append(f"<tr><td>audioread module</td><td>{b(ab.get('audioread_available'))}</td></tr>")
    rows.append(f"<tr><td>NIXPACKS_PKGS</td><td>{env.get('NIXPACKS_PKGS')}</td></tr>")
    rows.append(f"<tr><td>PYTHON_VERSION</td><td>{env.get('PYTHON_VERSION')}</td></tr>")

    imp_err_html = ""
    if imports:
        items = "".join(f"<li><code>{k}</code>: <code>{v}</code></li>" for k, v in imports.items())
        imp_err_html = f"<details><summary><strong>Import errors</strong></summary><ul class='mono'>{items}</ul></details>"

    np_conf = diag.get("numpy_config_snippet")
    np_html = f"<details><summary>NumPy config</summary><pre class='mono'>{np_conf}</pre></details>" if np_conf else ""

    recs = diag.get("recommendations") or []
    rec_html = "".join(f"<li>{r}</li>" for r in recs)

    return f"""
    <section class="box">
      <h4>Environment Diagnostics</h4>
      <table>
        <tbody>
          {''.join(rows)}
        </tbody>
      </table>
      {imp_err_html}
      {np_html}
      <details><summary>Raw JSON</summary><pre class="mono" style="white-space:pre-wrap;">{json.dumps(diag, indent=2)}</pre></details>
      <h5>Recommendations</h5>
      <ul>{rec_html}</ul>
    </section>
    """

# ---------------- templates ----------------

INDEX_HTML = """
<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Flash-cut Builder</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@picocss/pico@2/css/pico.min.css">
<style>
body{padding-block:1rem}
.grid{display:grid;gap:1rem;grid-template-columns:repeat(auto-fit,minmax(280px,1fr))}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
.wave{max-width:100%;border-radius:12px;border:1px solid #ddd;background:#fff}
.box{border:1px solid #ddd;border-radius:12px;padding:1rem;background:#fff}
.mobile-warning{background:#fef3c7;border:1px solid #f59e0b;padding:1rem;border-radius:8px;margin:1rem 0}
.loading{display:none;text-align:center;padding:2rem}
.spinner{border:4px solid #f3f3f3;border-top:4px solid #3498db;border-radius:50%;width:40px;height:40px;animation:spin 1s linear infinite;margin:0 auto 1rem}
@keyframes spin{0%{transform:rotate(0deg)}100%{transform:rotate(360deg)}}
</style>
<script>
function showLoading() {
  document.getElementById('loading').style.display = 'block';
  document.getElementById('form').style.display = 'none';
}
</script>
</head><body><main class="container">
<h2>Flash-cut Builder</h2>
<p class="mono"><a href="{{ url_for('diag_page') }}">Open full diagnostics → /diag</a></p>
{% with messages = get_flashed_messages() %}{% if messages %}<article>{% for m in messages %}<p>{{m}}</p>{% endfor %}</article>{% endif %}{% endwith %}

{{ diag_html|safe }}

<div id="loading" class="loading">
  <div class="spinner"></div>
  <p>Processing audio file... This may take a moment on mobile devices.</p>
  <p><small>Mobile devices use optimized settings for better performance.</small></p>
</div>

<form id="form" action="{{ url_for('analyze') }}" method="post" enctype="multipart/form-data" onsubmit="showLoading()">
  <fieldset><legend>Audio</legend>
    <input type="file" name="audio" accept=".mp3,.m4a,.wav,.aac,.ogg,.flac,audio/mpeg,audio/mp4,audio/x-m4a,audio/wav,audio/*" required>
  </fieldset>
  <div class="grid">
    <fieldset><legend>Analysis</legend>
      <label>FPS <input type="number" name="fps" step="1" min="10" max="120" value="30"></label>
      <label>Onset threshold <input type="number" name="threshold" step="0.01" min="0" max="1" value="0.30"></label>
      <label>Max gap (s) <input type="number" name="max_gap" step="0.05" min="0.1" max="10" value="5.0"></label>
    </fieldset>
    <fieldset><legend>Flash window</legend>
      <label>Start (s) <input type="number" name="flash_start" step="0.1" value="10"></label>
      <label>End (s) <input type="number" name="flash_end" step="0.1" value="25"></label>
      <label>Min flash gap (s) <input type="number" name="flash_gap" step="0.01" value="0.12"></label>
    </fieldset>
    <fieldset><legend>Render (optional)</legend>
      <label><input type="checkbox" name="do_render" value="1"> Render video with ffmpeg</label>
      <label>Video clips (multiple) <input type="file" name="videos" accept=".mp4,.mov,.mkv,.m4v,.webm,video/*" multiple></label>
      <label>PNG images (multiple) <input type="file" name="images" accept="image/png" multiple></label>
      <label>Clip portion <select name="clip_mode"><option value="head" selected>Head (start)</option><option value="tail">Tail (end)</option></select></label>
      <label>Output file name <input type="text" name="output_name" value="final_video.mp4"></label>
    </fieldset>
  </div>
  <button type="submit">Analyze</button>
</form>
</main></body></html>
"""

RESULT_HTML = """
<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Flash-cut Result</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@picocss/pico@2/css/pico.min.css">
<style>
.grid{display:grid;gap:1rem;grid-template-columns:repeat(auto-fit,minmax(280px,1fr))}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
.box{border:1px solid #ddd;border-radius:12px;padding:1rem;background:#fff}
img.wave{max-width:100%;border-radius:12px;border:1px solid #ddd;background:#fff;max-height:320px;object-fit:cover}
</style></head><body><main class="container">
<h2>Flash-cut Result</h2>
<p class="mono">Job: {{ job_id }} • <a href="{{ url_for('diag_page') }}">Diagnostics</a></p>
<div class="grid">
<section class="box">
  <h4>Summary</h4><ul>
    <li>Onsets: <strong>{{ num_onsets }}</strong></li>
    <li>Segments: <strong>{{ num_segments }}</strong></li>
    <li>Flash cuts: <strong>{{ num_flash }}</strong> ({{ flash_start }}–{{ flash_end }} s)</li>
    <li>FPS: {{ fps }}, Threshold: {{ threshold }}, Max gap: {{ max_gap }}</li>
  </ul>
  <div>
    <a href="{{ url_for('download', job_id=job_id, filename='cuts.json') }}">Download cuts.json</a> •
    <a href="{{ url_for('download', job_id=job_id, filename='cuts.csv') }}">Download cuts.csv</a> •
    <a href="{{ url_for('download', job_id=job_id, filename='waveform.png') }}">Download waveform.png</a>
    {% if rendered %} • <a href="{{ url_for('download', job_id=job_id, filename=output_name) }}">Download {{ output_name }}</a>{% endif %}
  </div>
</section>
<section class="box">
  <h4>Waveform</h4>
  <img class="wave" src="{{ url_for('download', job_id=job_id, filename='waveform.png') }}" alt="waveform">
</section>
</div>

{{ diag_html|safe }}

<details><summary>Segments (first 100)</summary><pre class="mono" style="white-space:pre-wrap">{{ segments_preview }}</pre></details>
<p><a href="{{ url_for('index') }}">← New analysis</a></p>
</main></body></html>
"""

# ---------------- core helpers ----------------

def run_cmd(cmd: List[str], cwd: str | Path | None = None):
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "FFmpeg failed (exit %s)\nCMD: %s\n--- STDOUT ---\n%s\n--- STDERR ---\n%s" % (
                proc.returncode, " ".join(cmd), proc.stdout or "", proc.stderr or ""
            )
        )
    return proc

def quantize_to_fps(times: List[float], fps: float) -> List[float]:
    return [round(t * fps) / fps for t in times]

def confidence_from_envelope(times, env, sr, hop):
    if len(times) == 0:
        return []
    frames = librosa.time_to_frames(times, sr=sr, hop_length=hop) if librosa else np.array([], dtype=int)
    frames = np.clip(frames, 0, len(env) - 1)
    vals = env[frames] if len(frames) else np.array([], dtype=float)
    scale = np.quantile(env, 0.98) or (env.max() or 1.0)
    return np.clip(vals / (scale if scale > 0 else 1.0), 0.0, 1.0).tolist()

def detect_onsets_flux(y: np.ndarray, sr: int, hop: int = FRAME_HOP, threshold: float = 0.30) -> List[dict]:
    if librosa is None:
        return []
    
    start_time = time.time()
    _, y_perc = librosa.effects.hpss(y)
    env = librosa.onset.onset_strength(y=y_perc, sr=sr, hop_length=hop, aggregate=np.median)
    onset_times = librosa.onset.onset_detect(
        onset_envelope=env, sr=sr, hop_length=hop, units="time",
        backtrack=False, pre_max=3, post_max=3, pre_avg=3, post_avg=3, delta=0.0
    )
    conf = confidence_from_envelope(onset_times, env, sr, hop)
    keep = [i for i, c in enumerate(conf) if c >= threshold]
    
    # Log performance
    duration = time.time() - start_time
    log_performance("onset_detection", duration)
    
    return [{"time": float(onset_times[i]), "confidence": float(conf[i])} for i in keep]

def detect_beats(audio_path: str, threshold: float):
    if librosa is None:
        raise RuntimeError("librosa not available on server — see Diagnostics below.")
    
    start_time = time.time()
    config = get_processing_config()
    
    # Let librosa choose best backend (soundfile -> libsndfile, fallback to audioread/ffmpeg)
    try:
        y, sr = librosa.load(audio_path, sr=config['sample_rate'], mono=True)
    except Exception as e:
        raise RuntimeError(
            "Audio decode failed. Ensure ffmpeg & libsndfile are installed on the server "
            "or upload WAV/FLAC/OGG. Original error: %s" % e
        )
    
    duration = float(librosa.get_duration(y=y, sr=sr))
    
    # Check duration limits for mobile
    if config['device_type'] == 'mobile' and duration > config['max_duration']:
        raise RuntimeError(
            f"Audio file too long for mobile device ({duration:.1f}s > {config['max_duration']:.1f}s). "
            f"Please upload a shorter file or use a desktop browser for longer files."
        )
    
    events = detect_onsets_flux(y, sr, hop=config['hop_length'], threshold=threshold)
    
    # Log total processing time
    total_duration = time.time() - start_time
    log_performance("audio_analysis_total", total_duration)
    
    return events, duration, sr, y

def compute_intervals(beats: List[dict], duration: float, fps: float, max_gap: float):
    end = round(duration * fps) / fps
    if not beats:
        splits = [0.0]; prev = 0.0
        while end - prev > max_gap:
            prev += max_gap; splits.append(prev)
        splits.append(end)
        starts = [round(s, 3) for s in splits[:-1]]
        ends = [round(e, 3) for e in splits[1:]]
        return starts, ends
    beat_times = quantize_to_fps(sorted(float(b["time"]) for b in beats), fps)
    splits = [0.0]; prev = 0.0
    first = beat_times[0]
    while first - prev > max_gap:
        prev += max_gap; splits.append(prev)
    splits.append(first)
    for i in range(1, len(beat_times)):
        L, R = beat_times[i - 1], beat_times[i]
        prev = L
        while R - prev > max_gap:
            prev += max_gap; splits.append(prev)
        splits.append(R)
    if end > splits[-1]:
        prev = splits[-1]
        while end - prev > max_gap:
            prev += max_gap; splits.append(prev)
        splits.append(end)
    starts = [round(s, 3) for s in splits[:-1]]
    ends = [round(e, 3) for e in splits[1:]]
    return starts, ends

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
            pruned.append(t); last = t
    return quantize_to_fps(pruned, fps)

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
        w = h = None; dur = None
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

# ---------------- routes ----------------

@app.route("/")
def index():
    diag = get_diag()
    return render_template_string(INDEX_HTML, diag_html=render_diag_html(diag))

@app.route("/diag")
def diag_page():
    diag = get_diag()
    PAGE = """
    <!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>Diagnostics</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@picocss/pico@2/css/pico.min.css">
    <style>.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}.box{border:1px solid #ddd;border-radius:12px;padding:1rem;background:#fff;}</style>
    </head><body><main class="container">
    <h2>Diagnostics</h2>
    <p><a href="{{ url_for('index') }}">← Back</a></p>
    {{ diag_html|safe }}
    </main></body></html>
    """
    return render_template_string(PAGE, diag_html=render_diag_html(diag))

@app.route("/analyze", methods=["POST"])
def analyze():
    start_time = time.time()
    config = get_processing_config()
    
    audio_file = request.files.get("audio")
    if not audio_file or audio_file.filename == "":
        flash("Please upload an audio file.")
        return redirect(url_for("index"))

    # Check file size for mobile devices
    if config['device_type'] == 'mobile':
        # Get file size (approximation from content length)
        file_size_mb = len(audio_file.read()) / (1024 * 1024)
        audio_file.seek(0)  # Reset file pointer
        
        if file_size_mb > 10:  # 10MB limit for mobile
            flash(f"File too large for mobile device ({file_size_mb:.1f}MB > 10MB). Please use a smaller file or desktop browser.")
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

    # Disable rendering for mobile to save resources
    if config['device_type'] == 'mobile' and do_render:
        flash("Video rendering disabled on mobile devices for better performance. Analysis only will be performed.")
        do_render = False

    job_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    audio_path = job_dir / audio_file.filename
    audio_file.save(str(audio_path))

    try:
        events, duration, sr, y = detect_beats(str(audio_path), threshold=threshold)
    except Exception as e:
        error_msg = str(e)
        if config['device_type'] == 'mobile':
            error_msg += "\n\nMobile devices have limited processing capabilities. Try a shorter file or use a desktop browser for larger files."
        flash(f"Failed to read audio. {error_msg}")
        return redirect(url_for("index"))

    starts, ends = compute_intervals(events, duration, fps, max_gap)

    try:
        flash_times = detect_flash_window(y, sr, (flash_start, flash_end), flash_gap, fps, threshold) if flash_end > flash_start else []
    except Exception as e:
        flash_times = []
        if config['device_type'] == 'mobile':
            # Don't fail completely on mobile, just skip flash detection
            flash("Flash detection skipped on mobile device to improve performance.")

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

    # Log total processing time
    total_time = time.time() - start_time
    log_performance("total_analysis", total_time)

    diag = get_diag()
    return render_template_string(
        RESULT_HTML,
        job_id=job_id,
        fps=fps, threshold=threshold, max_gap=max_gap,
        flash_start=flash_start, flash_end=flash_end,
        num_onsets=len(events), num_segments=len(starts), num_flash=len(flash_times),
        segments_preview=segments_preview,
        rendered=rendered, output_name=output_name,
        diag_html=render_diag_html(diag),
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
            out_s.append(s); out_e.append(e)
            continue
        pts = [s] + cuts + [e]
        for i in range(len(pts) - 1):
            a, b = pts[i], pts[i + 1]
            if b - a < 1.0 / fps:
                b = a + 1.0 / fps
            out_s.append(round(a, 3)); out_e.append(round(b, 3))
    return out_s, out_e

def plot_waveform(png_path: Path, y: np.ndarray, sr: int, flash_times: List[float], window: Tuple[float, float]):
    if plt is None or librosa is None:
        return
    
    start_time = time.time()
    config = get_processing_config()
    
    # Optimize plot for mobile devices
    if config['device_type'] == 'mobile':
        figsize = (12, 3)  # Smaller figure for mobile
        downsample = max(1, len(y) // 20000)  # Downsample for mobile
    else:
        figsize = (18, 4)
        downsample = max(1, len(y) // 50000)  # Less aggressive downsampling for desktop
    
    # Downsample data for faster rendering
    y_plot = y[::downsample]
    t = np.linspace(0, librosa.get_duration(y=y, sr=sr), num=len(y_plot), endpoint=True)
    
    plt.figure(figsize=figsize)
    plt.fill_between(t, y_plot, -y_plot, color="#f0b429", alpha=0.25)
    plt.plot(t, y_plot, color="#f0b429", lw=0.7, alpha=0.8)
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
    plt.savefig(png_path, dpi=config['plot_dpi'])
    plt.close()
    
    # Log plot generation time
    duration = time.time() - start_time
    log_performance("plot_generation", duration)

@app.route("/health")
def health():
    return {"ok": True}

@app.route("/logs")
def view_logs():
    """View performance logs for debugging mobile issues"""
    log_file = BASE_DIR / "logs" / "performance.log"
    if not log_file.exists():
        return "<h1>No performance logs available</h1>"
    
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()[-50:]  # Last 50 entries
        
        logs = []
        for line in lines:
            try:
                logs.append(json.loads(line.strip()))
            except:
                continue
        
        html = """
        <!doctype html><html><head><title>Performance Logs</title>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@picocss/pico@2/css/pico.min.css">
        <style>.mono{font-family:monospace}</style>
        </head><body><main class="container">
        <h2>Performance Logs (Last 50 entries)</h2>
        <p><a href="/">← Back to main</a></p>
        <table><thead><tr><th>Time</th><th>Operation</th><th>Duration (s)</th><th>Device</th><th>User Agent</th></tr></thead><tbody>
        """
        
        for log in reversed(logs):
            html += f"""<tr>
                <td class="mono">{log.get('timestamp', '')[:19]}</td>
                <td>{log.get('operation', '')}</td>
                <td>{log.get('duration_seconds', 0):.3f}</td>
                <td><span style="color:{'#f59e0b' if log.get('device_type') == 'mobile' else '#16a34a'}">{log.get('device_type', 'unknown')}</span></td>
                <td class="mono" style="font-size:0.8em">{log.get('user_agent', '')[:100]}</td>
            </tr>"""
        
        html += "</tbody></table></main></body></html>"
        return html
        
    except Exception as e:
        return f"<h1>Error reading logs: {e}</h1>"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)