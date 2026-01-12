# Pitch Glide Direction Threshold — Docker (Local)

Run this Streamlit app fully locally using Docker / Docker Compose.

## Quick start (stable / clinical use)

From the repository root (same folder as `pitch-glide-direction-threshold.py`):

```bash
docker compose up -d --build
```

Open:

- http://localhost:30000

Stop:

```bash
docker compose down
```

## Dev mode (hot-ish reload)

If you want edits to your local files to reflect inside the container:

```bash
docker compose -f docker-compose.dev.yml up -d --build
```

Open:

- http://localhost:30000

## Notes

- The container listens on port 8501 internally (Streamlit default).
- The host (macOS) binds to **localhost:30000** (not exposed to LAN).
- If you get a port conflict error, something else is already using 30000.
  Change the left side of the mapping in `docker-compose.yml`:
  `127.0.0.1:30000:8501` -> `127.0.0.1:<NEWPORT>:8501`
