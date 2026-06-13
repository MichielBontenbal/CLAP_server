Deploy the vibe_sound project to the Raspberry Pi over Tailscale and restart the systemd service.

## Connection
- Host: `pi@100.94.220.68` (Tailscale IP)
- Project path on Pi: `/home/pi/CLAP_server`

## Steps

Run these commands sequentially via SSH. Stop and report if any step fails.

1. **Pull latest code from GitHub**
```bash
ssh pi@100.94.220.68 "cd /home/pi/CLAP_server && git pull origin main"
```

2. **Sync dependencies with uv**
```bash
ssh pi@100.94.220.68 "cd /home/pi/CLAP_server && ~/.local/bin/uv sync"
```

3. **Restart the systemd service**
```bash
ssh pi@100.94.220.68 "sudo systemctl restart vibe-sound"
```

4. **Check service status**
```bash
ssh pi@100.94.220.68 "sudo systemctl status vibe-sound --no-pager"
```

5. **Show last 30 log lines**
```bash
ssh pi@100.94.220.68 "sudo journalctl -u vibe-sound -n 30 --no-pager"
```

## After deploy
Report whether the service is active, show any errors from the logs, and confirm the URL the Pi is serving on (port 8443 over Tailscale: `https://100.94.220.68:8443`).

## Notes
- The Pi uses `uv` at `~/.local/bin/uv` (same as the dev machine)
- The service runs uvicorn with SSL on port 8443 — certs live in `certs/` on the Pi
- Logs: `sudo journalctl -u vibe-sound -f` on the Pi for live output
- If `uv` is not found, fall back to `.venv/bin/pip install -r requirements.txt` or `pip install -e .`
