# open-med-mcp "lungmask" adapter (CUDA 12.4 runtime). Weights download on first use into /weights.
#   docker build -f src/open_med_mcp/zoo/lungmask/Dockerfile -t ghcr.io/d0ng231/open-med-mcp-lungmask:0.1 .
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 OMM_WEIGHTS_DIR=/weights TORCH_HOME=/weights
WORKDIR /app
RUN pip install --no-cache-dir "lungmask>=0.2.20" "SimpleITK>=2.3" "pillow>=9.4" "scipy>=1.10"
COPY src/open_med_mcp/zoo/_sdk/omm_job.py /app/omm_job.py
COPY src/open_med_mcp/zoo/lungmask/run.py /app/run.py
ENTRYPOINT ["python", "/app/run.py"]
CMD ["--job", "/job"]
