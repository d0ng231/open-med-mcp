# open-med-mcp "classical" adapter: thresholds / Otsu / region growing (CPU only, ~200 MB)
# Build from the repository root:
#   docker build -f src/open_med_mcp/zoo/classical/Dockerfile -t ghcr.io/d0ng231/open-med-mcp-classical:0.1 .
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN pip install --no-cache-dir "numpy>=1.24" "scipy>=1.10" "SimpleITK>=2.3" "pillow>=9.4"
WORKDIR /app
COPY src/open_med_mcp/zoo/_sdk/omm_job.py /app/omm_job.py
COPY src/open_med_mcp/zoo/classical/run.py /app/run.py
ENTRYPOINT ["python", "/app/run.py"]
CMD ["--job", "/job"]
