# Vibe Sound — CLAP urban sound classifier

A mobile-friendly web app that records audio on your phone, sends it to a FastAPI server, and classifies the sounds using [LAION CLAP](https://github.com/LAION-AI/CLAP) (`laion/larger_clap_general`) — a zero-shot audio classification model. Results and a mel spectrogram are shown immediately on the phone.

Built for urban sound research at [Marineterrein Amsterdam](https://marineterrein.nl), following the [urbansounds2025](https://github.com/sensemakersamsterdam/urbansounds2025) project.

## Features

- **Mobile recorder** — record up to 15 seconds of audio from your phone browser; audio is resampled to 48 kHz mono and posted to the server
- **Zero-shot classification** — no per-class training data needed; CLAP compares audio embeddings against text label lists
- **Three soundscapes** — City, Home, and Marineterrein, each with their own curated label set (`sound_scapes.py`)
- **Live mel spectrogram** — real-time scrolling waterfall streamed over WebSocket from the server microphone
- **Recording mel spectrogram** — static mel spectrogram rendered for each classified recording
- **Desktop dashboard** — recordings history, controls, and live spectrogram in a multi-iframe layout

## How it works

1. Open the mobile page on your phone (`/mobile`) over HTTPS
2. Select a soundscape (City / Home / Marineterrein)
3. Press **Record** — audio is captured in the browser (max 15 s)
4. Press **Classify** — the phone sends raw float32 PCM to `/classify-audio?soundscape=<name>` on the server
5. The server runs CLAP and returns the top-3 labels + a mel spectrogram PNG
6. Results appear on your phone instantly

## Run

```bash
uv run clap_server.py
```

The server starts on **port 8080**. For phone access it must be reachable over HTTPS — use the self-signed certificate in `certs/`:

```bash
# Regenerate the cert (e.g. after your LAN IP changes)
openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:prime256v1 -days 825 -nodes \
  -keyout certs/key.pem -out certs/cert.pem -subj "/CN=vibe-sound" \
  -addext "subjectAltName=DNS:localhost,IP:127.0.0.1,IP:$(ipconfig getifaddr en0)"
```

Alternatives for access beyond your LAN: `tailscale serve 8080` or `ngrok http 8080`.

## Routes

| Route | Description |
|---|---|
| `/` | Desktop dashboard (recordings + controls + live spectrogram) |
| `/mobile` | Mobile recorder page |
| `/info` | Information page (how it works, soundscape labels) |
| `/spectrogram` | Live mel spectrogram (WebSocket) |
| `/controls` | Recorder controls iframe |
| `/recordings` | Recordings list iframe |
| `/classify-audio?soundscape=` | POST float32 PCM → JSON results + spectrogram PNG |

## Soundscapes

Label lists are defined in `sound_scapes.py`:

- **City** — traffic, sirens, birds, drilling, tram, train, etc.
- **Home** — washing machine, footsteps, door slamming, etc.
- **Marineterrein** — water, birds, fountain, thunderstorm, wind, etc.

## Stack

- [FastAPI](https://fastapi.tiangolo.com) + [uvicorn](https://www.uvicorn.org)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers) — `zero-shot-audio-classification` pipeline
- [LAION CLAP](https://huggingface.co/laion/larger_clap_general) — `laion/larger_clap_general`
- NumPy / SciPy / Matplotlib for DSP and spectrogram rendering
- Plain HTML + vanilla JS on the frontend (no framework)

## DSP parameters

Spectrogram parameters live at the top of `live_spectrogram.py`: `N_FFT` (2048), `HOP` (512), `N_MELS` (80), `FMIN`, and `DB_RANGE` (70 dB auto-gained dynamic range).
