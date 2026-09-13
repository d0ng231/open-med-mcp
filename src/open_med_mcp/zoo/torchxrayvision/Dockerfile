# open-med-mcp "torchxrayvision" adapter (CPU is enough; ~1.5 GB with CPU torch)
#   docker build -f src/open_med_mcp/zoo/torchxrayvision/Dockerfile -t ghcr.io/d0ng231/open-med-mcp-torchxrayvision:0.1 .
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 OMM_WEIGHTS_DIR=/weights
WORKDIR /app
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch>=2.5" "torchvision>=0.20" \
    && pip install --no-cache-dir "torchxrayvision>=1.3" "SimpleITK>=2.3" "pillow>=9.4" "scikit-image>=0.22"
COPY src/open_med_mcp/zoo/_sdk/omm_job.py /app/omm_job.py
COPY src/open_med_mcp/zoo/torchxrayvision/run.py /app/run.py
ENTRYPOINT ["python", "/app/run.py"]
CMD ["--job", "/job"]
