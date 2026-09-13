# open-med-mcp "sam2" adapter: SAM 2.1 + MedSAM2 checkpoints (CUDA 12.4 runtime, ~7 GB)
# Build from the repository root:
#   docker build -f src/open_med_mcp/zoo/sam2/Dockerfile -t ghcr.io/d0ng231/open-med-mcp-sam2:0.1 .
# Weights are NOT baked in: mount them at /weights (open-med-mcp models download sam2).
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 SAM2_BUILD_CUDA=0 OMM_WEIGHTS_DIR=/weights
WORKDIR /app
COPY src/open_med_mcp/zoo/sam2/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY src/open_med_mcp/zoo/_sdk/omm_job.py /app/omm_job.py
COPY src/open_med_mcp/zoo/sam2/run.py /app/run.py
COPY src/open_med_mcp/zoo/sam2/configs /app/configs
ENTRYPOINT ["python", "/app/run.py"]
CMD ["--job", "/job"]
