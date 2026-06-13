"""Real-time mel-spectrogram backend.

The browser streams raw Float32 PCM over a WebSocket; this server computes
an STFT + mel filterbank with numpy and streams uint8 mel frames back.
"""

import json
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

N_FFT = 2048
HOP = 512
N_MELS = 80
FMIN = 30.0
DB_RANGE = 70.0  # dynamic range mapped onto the 0..255 colour scale
DB_OFFSET = 94.0  # rough dBFS -> dB SPL offset (uncalibrated, as in urbansounds2025)
DB_METER_MAX = 110.0  # top of the VU/peak meter scale

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Vibe Sound — real-time mel spectrogram")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(sr: float, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    """Triangular mel filterbank, shape (n_mels, n_fft//2 + 1)."""
    freqs = np.linspace(0.0, sr / 2.0, n_fft // 2 + 1)
    mel_pts = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))
    fb = np.zeros((n_mels, freqs.size), dtype=np.float32)
    for i in range(n_mels):
        lo, center, hi = mel_pts[i], mel_pts[i + 1], mel_pts[i + 2]
        rising = (freqs - lo) / (center - lo)
        falling = (hi - freqs) / (hi - center)
        fb[i] = np.maximum(0.0, np.minimum(rising, falling))
    return fb


class SpectrogramProcessor:
    def __init__(self, sample_rate: float):
        self.sr = sample_rate
        self.fmax = sample_rate / 2.0
        self.window = np.hanning(N_FFT).astype(np.float32)
        self.fb = mel_filterbank(sample_rate, N_FFT, N_MELS, FMIN, self.fmax)
        self.buffer = np.zeros(0, dtype=np.float32)
        self.ref_db = -30.0  # running reference level for auto-gain

    def process(self, samples: np.ndarray) -> bytes | None:
        """Consume PCM samples, return zero or more mel frames as uint8 bytes."""
        self.buffer = np.concatenate([self.buffer, samples])
        frames = []
        while self.buffer.size >= N_FFT:
            segment = self.buffer[:N_FFT] * self.window
            self.buffer = self.buffer[HOP:]
            power = np.abs(np.fft.rfft(segment)) ** 2
            mel = self.fb @ power
            frames.append(10.0 * np.log10(np.maximum(mel, 1e-10)))
        if not frames:
            return None
        db = np.stack(frames)
        # Track the loudest level seen: jump up instantly, decay slowly,
        # and never drop below -30 dB so silence doesn't pump the gain.
        self.ref_db = max(self.ref_db - 0.1 * len(frames), float(db.max()), -30.0)
        floor = self.ref_db - DB_RANGE
        scaled = np.clip((db - floor) / DB_RANGE * 255.0, 0.0, 255.0)
        return scaled.astype(np.uint8).tobytes()


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.websocket("/ws")
async def spectrogram_ws(websocket: WebSocket):
    await websocket.accept()
    try:
        hello = json.loads(await websocket.receive_text())
        processor = SpectrogramProcessor(float(hello["sampleRate"]))
        await websocket.send_text(json.dumps({
            "type": "config",
            "nMels": N_MELS,
            "nFft": N_FFT,
            "hop": HOP,
            "fmin": FMIN,
            "fmax": processor.fmax,
            "sampleRate": processor.sr,
            "meterMax": DB_METER_MAX,
        }))
        while True:
            data = await websocket.receive_bytes()
            pcm = np.frombuffer(data, dtype=np.float32)
            if pcm.size:
                rms = float(np.sqrt(np.mean(pcm * pcm)))
                level = 20.0 * np.log10(rms + 1e-10) + DB_OFFSET
                level = max(0.0, min(DB_METER_MAX, level))
                await websocket.send_text(json.dumps({"type": "level", "db": round(level, 1)}))
            out = processor.process(pcm)
            if out:
                await websocket.send_bytes(out)
    except WebSocketDisconnect:
        pass


def lan_ip() -> str:
    """Best-effort LAN IP, for printing the URL to open from a phone."""
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))  # no traffic sent; just selects an interface
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


if __name__ == "__main__":
    import uvicorn

    cert = Path(__file__).parent / "certs" / "cert.pem"
    key = Path(__file__).parent / "certs" / "key.pem"
    if cert.exists() and key.exists():
        # HTTPS: required for microphone access from other devices (getUserMedia
        # only works in a secure context). Self-signed, so the browser shows a
        # one-time warning to accept.
        print(f"\n  On this machine:  https://localhost:8443")
        print(f"  From your phone:  https://{lan_ip()}:8443  (same WiFi, accept the cert warning)\n")
        uvicorn.run(
            "live_spectrogram:app",
            host="0.0.0.0",
            port=8443,
            reload=True,
            ssl_certfile=str(cert),
            ssl_keyfile=str(key),
        )
    else:
        print("\n  No certs/ found — serving plain HTTP (mic only works on localhost)")
        print("  http://localhost:8000\n")
        uvicorn.run("live_spectrogram:app", host="0.0.0.0", port=8000, reload=True)
