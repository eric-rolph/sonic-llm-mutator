# Linux training environment: Python 3.12 + stable-retro.
#
# Build:
#   docker build -t sonic-llm-mutator .
#
# Train. This pipeline runs model-generated code, so the container is a
# security boundary, not just packaging -- the image runs as a non-root user
# and the recommended run drops capabilities and writable surface:
#   docker run --rm --cap-drop=ALL --read-only --tmpfs /tmp \
#       -v "$PWD/artifacts:/app/artifacts" -v "$PWD/policies:/app/policies" \
#       -v /path/to/roms:/roms:ro --env-file .env \
#       --add-host host.docker.internal:host-gateway sonic-llm-mutator \
#       sh -c "python -m stable_retro.import /roms && python main.py"
# Point MICRO_BASE_URL/MACRO_BASE_URL at a reachable model server (e.g.
# http://host.docker.internal:1234/v1 for LM Studio on the host); add
# --network=none instead of --add-host when the model endpoint is in-container.

FROM python:3.12-slim

WORKDIR /app

COPY requirements-linux.txt ./
RUN pip install --no-cache-dir -r requirements-linux.txt

COPY . .

# Run as an unprivileged user: a sandbox escape from generated code should not
# land as root, and the writable surface is limited to mounted volumes.
RUN useradd --create-home --uid 10001 sonic \
    && mkdir -p /app/artifacts /app/policies \
    && chown -R sonic:sonic /app
USER sonic

ENV SONIC_RETRO_BACKEND=stable

CMD ["python", "main.py"]
