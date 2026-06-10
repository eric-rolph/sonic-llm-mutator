# Linux training environment: Python 3.12 + stable-retro.
#
# Build:
#   docker build -t sonic-llm-mutator .
#
# Train. Mount the repo (so evolved policies/artifacts persist on the host)
# and your legally obtained ROMs; the ROM import is per-container, so chain it
# before main.py. Point MICRO_BASE_URL/MACRO_BASE_URL at a model server
# reachable from the container (e.g. http://host.docker.internal:1234/v1 for
# LM Studio on the host):
#   docker run --rm -v "$PWD:/app" -v /path/to/roms:/roms --env-file .env \
#       --add-host host.docker.internal:host-gateway sonic-llm-mutator \
#       sh -c "python -m stable_retro.import /roms && python main.py"

FROM python:3.12-slim

WORKDIR /app

COPY requirements-linux.txt ./
RUN pip install --no-cache-dir -r requirements-linux.txt

COPY . .

ENV SONIC_RETRO_BACKEND=stable

CMD ["python", "main.py"]
