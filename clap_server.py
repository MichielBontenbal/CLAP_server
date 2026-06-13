"""FastAPI server for zero-shot audio classification with CLAP.

Model and labels follow urbansounds2025
(https://github.com/sensemakersamsterdam/urbansounds2025):
laion/larger_clap_general with the marineterrein candidate labels from
sound_scapes.py.

The frontend is plain HTML rendered from Python — the buttons are form posts,
no JavaScript. Recording happens server-side from this machine's microphone.
"""

import base64
import datetime
import io
import threading
import time
from pathlib import Path

import numpy as np
import scipy.io.wavfile
import sounddevice as sd
from fastapi import FastAPI, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

import live_spectrogram
import sound_scapes

MODEL_NAME = "laion/larger_clap_general"
SAMPLE_RATE = 48000  # CLAP's expected sampling rate
CANDIDATE_LABELS = sound_scapes.city_soundscape
TOP_N = 3
RECORDINGS_DIR = Path(__file__).parent / "recordings"

try:
    sd.query_devices(kind="input")
    HAS_AUDIO_INPUT = True
except Exception:
    HAS_AUDIO_INPUT = False

app = FastAPI(title="CLAP urban sound classifier")

# Live spectrogram, reused from live_spectrogram.py: register its WebSocket,
# static files and page on this app so the embedded view works unchanged.
app.add_api_websocket_route("/ws", live_spectrogram.spectrogram_ws)
app.mount("/static", StaticFiles(directory=live_spectrogram.STATIC_DIR), name="static")
app.add_api_route("/spectrogram", live_spectrogram.index, methods=["GET"])


class Recorder:
    """Records mono float32 audio from the default input device."""

    def __init__(self):
        self.stream = None
        self.chunks = []
        self.started_at = None
        self.lock = threading.Lock()

    @property
    def recording(self) -> bool:
        return self.stream is not None

    def start(self):
        self.chunks = []
        self.started_at = datetime.datetime.now()

        def callback(indata, frames, time, status):
            with self.lock:
                self.chunks.append(indata.copy())

        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=callback
        )
        self.stream.start()

    def stop(self) -> np.ndarray:
        self.stream.stop()
        self.stream.close()
        self.stream = None
        with self.lock:
            if not self.chunks:
                return np.zeros(0, dtype="float32")
            return np.concatenate(self.chunks).flatten()

    def elapsed(self) -> float:
        return (datetime.datetime.now() - self.started_at).total_seconds()


recorder = Recorder()
audio_data: np.ndarray | None = None
results: list[dict] | None = None
spectrogram_png: bytes | None = None
spectrogram_version = 0
message = "Ready — press Start recording."
recordings_log: list[dict] = []   # {filename, duration, started_at, results}
current_recording_filename: str | None = None



def render_spectrogram_png(audio: np.ndarray) -> bytes | None:
    """Mel spectrogram of a recording as a standalone PNG."""
    from matplotlib.figure import Figure

    processor = live_spectrogram.SpectrogramProcessor(SAMPLE_RATE)
    frames = processor.process(audio)
    if frames is None:
        return None
    mel = np.frombuffer(frames, dtype=np.uint8).reshape(-1, live_spectrogram.N_MELS)
    duration = mel.shape[0] * live_spectrogram.HOP / SAMPLE_RATE

    fig = Figure(figsize=(8, 2.4), dpi=120)
    fig.patch.set_facecolor("#0d1017")
    ax = fig.subplots()
    ax.imshow(
        mel.T, origin="lower", aspect="auto", cmap="viridis",
        vmin=0, vmax=255, extent=(0.0, duration, 0.0, live_spectrogram.N_MELS),
    )
    mel_lo = live_spectrogram.hz_to_mel(live_spectrogram.FMIN)
    mel_hi = live_spectrogram.hz_to_mel(processor.fmax)
    ticks, tick_labels = [], []
    for hz in (100, 250, 500, 1000, 2000, 4000, 8000, 16000):
        if live_spectrogram.FMIN <= hz <= processor.fmax:
            frac = (live_spectrogram.hz_to_mel(hz) - mel_lo) / (mel_hi - mel_lo)
            ticks.append(float(frac) * live_spectrogram.N_MELS)
            tick_labels.append(f"{hz // 1000} kHz" if hz >= 1000 else f"{hz} Hz")
    ax.set_yticks(ticks, tick_labels)
    ax.tick_params(colors="#8b93a3")
    ax.set_xlabel("time (s)", color="#8b93a3")
    for spine in ax.spines.values():
        spine.set_visible(False)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    return buf.getvalue()



_classifier = None
_classifier_lock = threading.Lock()


def get_classifier():
    """Initialize the zero-shot audio classification model (lazily, once)."""
    global _classifier
    with _classifier_lock:
        if _classifier is None:
            from transformers import pipeline

            _classifier = pipeline(
                task="zero-shot-audio-classification", model=MODEL_NAME
            )
    return _classifier


@app.on_event("startup")
def preload_model():
    RECORDINGS_DIR.mkdir(exist_ok=True)
    for wav in sorted(RECORDINGS_DIR.glob("recording_*.wav")):
        try:
            sr, data = scipy.io.wavfile.read(wav)
            recordings_log.append({
                "filename": wav.name,
                "duration": len(data) / sr,
                "started_at": None,
                "results": None,
            })
        except Exception:
            pass
    # Load CLAP in the background so the first Classify click doesn't stall.
    threading.Thread(target=get_classifier, daemon=True).start()


@app.post("/start")
def start_recording():
    global message, results, spectrogram_png
    if not HAS_AUDIO_INPUT:
        message = "No audio input device on this machine. Upload a recording via the mobile app."
        return RedirectResponse("/controls", status_code=303)
    if not recorder.recording:
        recorder.start()
        results = None
        spectrogram_png = None
        message = "Recording…"
    return RedirectResponse("/controls", status_code=303)


@app.post("/stop")
def stop_recording():
    global audio_data, message, current_recording_filename
    if recorder.recording:
        started = recorder.started_at
        audio_data = recorder.stop()
        duration = len(audio_data) / SAMPLE_RATE
        message = f"Recorded {duration:.1f} s — press Classify."
        filename = f"recording_{started.strftime('%Y%m%d_%H%M%S')}.wav"
        scipy.io.wavfile.write(RECORDINGS_DIR / filename, SAMPLE_RATE, audio_data)
        current_recording_filename = filename
        recordings_log.append({
            "filename": filename,
            "duration": duration,
            "started_at": started,
            "results": None,
        })
    return RedirectResponse("/controls", status_code=303)


@app.post("/play")
def play_recording():
    global message
    if audio_data is None or audio_data.size == 0:
        message = "Nothing recorded yet — record some audio first."
    else:
        sd.play(audio_data, SAMPLE_RATE)  # non-blocking, plays on the server's speakers
        message = f"Playing {len(audio_data) / SAMPLE_RATE:.1f} s recording…"
    return RedirectResponse("/controls", status_code=303)


def _run_classify(audio_snapshot: np.ndarray) -> None:
    """Run CLAP pipeline + render spectrogram."""
    global results, message, spectrogram_png, spectrogram_version

    output = get_classifier()(audio_snapshot, candidate_labels=CANDIDATE_LABELS)
    results = output[:TOP_N]
    duration = len(audio_snapshot) / SAMPLE_RATE
    message = f"Classified {duration:.1f} s of audio."

    spectrogram_png = render_spectrogram_png(audio_snapshot)
    spectrogram_version += 1

    for rec in recordings_log:
        if rec["filename"] == current_recording_filename:
            rec["results"] = results
            break


@app.post("/classify")
def classify():
    global message
    if audio_data is None or audio_data.size == 0:
        message = "Nothing recorded yet — record some audio first."
        return RedirectResponse("/controls", status_code=303)
    _run_classify(audio_data.copy())
    return RedirectResponse("/controls", status_code=303)



@app.post("/load-classify/{filename}")
def load_classify(filename: str):
    global audio_data, current_recording_filename, results, spectrogram_png, message
    safe = (RECORDINGS_DIR / filename).resolve()
    if not str(safe).startswith(str(RECORDINGS_DIR.resolve())) or not safe.exists():
        message = "Recording not found."
        return RedirectResponse("/controls", status_code=303)
    _, data = scipy.io.wavfile.read(safe)
    audio_data = data.astype(np.float32)
    current_recording_filename = filename
    results = None
    spectrogram_png = None
    _run_classify(audio_data.copy())
    return RedirectResponse("/controls", status_code=303)


@app.get("/recording-spectrogram.png")
def recording_spectrogram():
    if spectrogram_png is None:
        return Response(status_code=404)
    return Response(content=spectrogram_png, media_type="image/png")



@app.get("/recordings", response_class=HTMLResponse)
def recordings_page():
    def rec_ts(rec: dict) -> tuple[str, str]:
        dt = rec["started_at"]
        if dt is None:
            try:
                dt = datetime.datetime.strptime(rec["filename"][10:25], "%Y%m%d_%H%M%S")
            except Exception:
                return ("", rec["filename"])
        return (dt.strftime("%d %b"), dt.strftime("%H:%M:%S"))

    items = ""
    for rec in reversed(recordings_log):
        active = " active" if rec["filename"] == current_recording_filename else ""
        date, time_ = rec_ts(rec)
        items += (
            f'<div class="rec{active}" data-file="{rec["filename"]}">'
            f'<div class="time">{time_}</div>'
            f'<div class="meta">{date} &middot; {rec["duration"]:.1f}s</div>'
            f'</div>'
        )
    if not items:
        items = '<p class="empty">No recordings yet.</p>'
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="5">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, "Segoe UI", sans-serif; background: #0d1017;
         color: #d7dce5; padding: 0.5rem; }}
  h2 {{ font-size: 0.68rem; font-weight: 600; color: #5b6373; margin-bottom: 0.5rem;
       text-transform: uppercase; letter-spacing: 0.07em; padding: 0 0.25rem; }}
  .rec {{ padding: 0.4rem 0.5rem; border-radius: 5px; cursor: pointer;
          border: 1px solid transparent; margin-bottom: 0.25rem; user-select: none; }}
  .rec:hover {{ background: #141d2e; border-color: #1e2840; }}
  .rec.active {{ border-color: #36c46d; }}
  .time {{ font-size: 0.8rem; color: #d7dce5; font-variant-numeric: tabular-nums; }}
  .meta {{ font-size: 0.68rem; color: #5b6373; margin-top: 0.1rem; }}
  .rec.active .time {{ color: #4ac16d; }}
  .empty {{ color: #5b6373; font-size: 0.78rem; padding: 0.4rem; }}
</style>
</head>
<body>
<h2>Recordings</h2>
{items}
<script>
document.querySelectorAll('.rec').forEach(function(el) {{
  el.addEventListener('dblclick', function() {{
    var form = document.createElement('form');
    form.method = 'post';
    form.action = '/load-classify/' + el.dataset.file;
    form.target = 'controls-frame';
    document.body.appendChild(form);
    form.submit();
  }});
}});
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    # Wrapper page. Controls and spectrogram live in separate iframes so that
    # button posts (and the recording counter refresh) only reload the controls
    # frame, leaving the live spectrogram running.
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CLAP sound classifier</title>
<style>
  body {{ font-family: -apple-system, "Segoe UI", sans-serif; background: #0d1017; color: #d7dce5;
         max-width: 980px; margin: 3rem auto; padding: 0 1rem; }}
  h1 {{ font-size: 1.2rem; margin-bottom: 0.75rem; }} h1 small {{ color: #8b93a3; font-weight: 400; }}
  .layout {{ display: flex; gap: 1rem; align-items: flex-start; }}
  iframe {{ border: 0; display: block; }}
  #recordings-frame {{ width: 195px; flex: none; height: 760px; border-radius: 8px; }}
  .main {{ flex: 1; min-width: 0; }}
  .main iframe {{ width: 100%; }}
  #controls-frame {{ height: 720px; }}
  #spectrogram-frame {{ height: 480px; border-radius: 8px; margin-top: 1rem; }}
</style>
</head>
<body>
<h1>CLAP sound classifier <small>— {MODEL_NAME}</small></h1>
<div class="layout">
  <iframe id="recordings-frame" src="/recordings" title="saved recordings"></iframe>
  <div class="main">
    <iframe id="controls-frame" name="controls-frame" src="/controls" title="recorder controls"></iframe>
    <iframe id="spectrogram-frame" src="/spectrogram" allow="microphone" title="live spectrogram"></iframe>
  </div>
</div>
</body>
</html>"""


@app.get("/controls", response_class=HTMLResponse)
def controls():
    if recorder.recording:
        refresh = '<meta http-equiv="refresh" content="1">'
    else:
        refresh = ""
    if recorder.recording:
        status = f"&#x1F534; recording… {recorder.elapsed():.0f} s"
    else:
        status = message

    def button(action: str, label: str, enabled: bool, accent: str = "") -> str:
        disabled = "" if enabled else "disabled"
        return (
            f'<form method="post" action="{action}">'
            f'<button class="{accent}" {disabled}>{label}</button></form>'
        )

    if HAS_AUDIO_INPUT:
        rec_buttons = (
            button("/start", "Start recording", not recorder.recording, "rec")
            + button("/stop", "End recording", recorder.recording)
        )
    else:
        rec_buttons = '<p style="color:#888;font-size:0.85rem;margin:0">No mic on this device — use the <a href="/mobile" target="_top" style="color:#7ec8e3">mobile app</a> to send audio.</p>'
    action_buttons = (
        button("/play", "Play recording", not recorder.recording and audio_data is not None)
        + button("/classify", "Classify", not recorder.recording and audio_data is not None, "go")
    )

    rows = ""
    if results:
        table_rows = ""
        for r in results:
            pct = r["score"] * 100
            table_rows += (
                f'<tr><td>{r["label"]}</td><td class="score">{r["score"]:.3f}</td>'
                f'<td class="barcell"><div class="bar" style="width:{pct:.1f}%"></div></td></tr>'
            )
        rows = f'<h3 class="sh">Classification</h3><table><tr><th>label</th><th>score</th><th></th></tr>{table_rows}</table>'
        if spectrogram_png is not None:
            rows += (
                f'<h3 class="sh">Mel spectrogram</h3>'
                f'<img src="/recording-spectrogram.png?v={spectrogram_version}" alt="mel spectrogram">'
            )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
{refresh}
<title>controls</title>
<style>
  body {{ font-family: -apple-system, "Segoe UI", sans-serif; background: #0d1017; color: #d7dce5;
         margin: 0; }}
  .panel {{ border: 1px solid #1e2840; border-radius: 8px; padding: 0.7rem 0.9rem; margin-bottom: 0.75rem; }}
  .panel-title {{ font-size: 0.68rem; font-weight: 600; color: #5b6373; text-transform: uppercase;
                  letter-spacing: 0.07em; margin-bottom: 0.55rem; }}
  #status {{ color: #8b93a3; margin: 0.35rem 0 0; min-height: 1.2em; font-size: 0.85rem; }}
  .controls {{ display: flex; gap: 0.6rem; }}
  form {{ display: inline; }}
  button {{ font: inherit; padding: 0.55rem 1.2rem; border-radius: 8px; cursor: pointer;
            border: 1px solid #3a4254; background: #1c2333; color: #e8ecf3; }}
  button:disabled {{ opacity: 0.35; cursor: default; }}
  button.rec:enabled {{ background: #c43636; border-color: #c43636; }}
  button.go:enabled {{ background: #36c46d; border-color: #36c46d; color: #06200f; font-weight: 600; }}
  table {{ width: 100%; margin-top: 1.2rem; border-collapse: collapse; }}
  th {{ text-align: left; color: #8b93a3; font-weight: 500; font-size: 0.8rem; }}
  td, th {{ padding: 0.4rem 0.6rem 0.4rem 0; }}
  .score {{ font-variant-numeric: tabular-nums; }}
  .barcell {{ width: 50%; }}
  .bar {{ height: 0.8rem; border-radius: 4px; background: linear-gradient(90deg, #277f8e, #4ac16d); }}
  .caption {{ color: #8b93a3; font-size: 0.8rem; margin: 0.5rem 0; }}
  .sh {{ font-size: 0.72rem; font-weight: 600; color: #5b6373; margin: 1.1rem 0 0.4rem;
         text-transform: uppercase; letter-spacing: 0.06em; }}
  img {{ width: 100%; height: auto; border-radius: 6px; }}
</style>
</head>
<body>
<div class="panel">
  <div class="panel-title">New recording</div>
  <div class="controls">{rec_buttons}</div>
  <div id="status">{status}</div>
</div>
<div class="controls">{action_buttons}</div>
{rows}
</body>
</html>"""


@app.get("/mobile")
async def mobile():
    return FileResponse(live_spectrogram.STATIC_DIR / "mobile.html")


@app.get("/info")
async def info():
    return FileResponse(live_spectrogram.STATIC_DIR / "docs.html")


@app.post("/classify-audio")
async def classify_audio(request: Request, soundscape: str = Query(default="city")):
    """Accept raw float32 mono PCM at 48 kHz, return classification + spectrogram PNG."""
    body = await request.body()
    if not body:
        return Response(status_code=400)
    if soundscape == "home":
        labels = sound_scapes.home_soundscape
    elif soundscape == "marineterrein":
        labels = sound_scapes.marineterrein_soundscape
    else:
        labels = sound_scapes.city_soundscape
    pcm = np.frombuffer(body, dtype=np.float32).copy()
    output = get_classifier()(pcm, candidate_labels=labels)
    top = [{"label": r["label"], "score": float(r["score"])} for r in output[:TOP_N]]
    spec = render_spectrogram_png(pcm)
    spec_b64 = base64.b64encode(spec).decode() if spec else None
    return {"results": top, "spectrogram": spec_b64}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("clap_server:app", host="0.0.0.0", port=8080, reload=True)
