# --- Base image ---
# Slim Python image: has just enough OS + Python, keeps the final image small
FROM python:3.11-slim

# --- Working directory inside the container ---
WORKDIR /app

# --- Install dependencies first (before copying code) ---
# Docker caches each step. If only your code changes (not requirements.txt),
# this layer is reused instead of reinstalling everything — much faster
# rebuilds during development.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Copy the rest of the backend code ---
COPY . .

# --- Document which port the app listens on (informational; doesn't
# actually publish it — that's done with `-p` on `docker run`) ---
EXPOSE 8000

# --- Container startup command ---
# Uses shell form (not exec-array form) so $PORT gets expanded — Railway
# (and most PaaS platforms) inject their own PORT env var at runtime and
# route traffic to whatever port the app actually binds to. Falls back
# to 8000 for local `docker run` / docker-compose where PORT isn't set.
CMD sh -c "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"
