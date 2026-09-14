# open-med-mcp "voxtell" adapter (CUDA 12.6 runtime, PyTorch 2.8 - VoxTell requires torch < 2.9).
# Checkpoint (1.7 GB) and the Qwen3-Embedding-4B text encoder (~8 GB) download on first use into /weights.
#   docker build -f src/open_med_mcp/zoo/voxtell/Dockerfile -t ghcr.io/d0ng231/open-med-mcp-voxtell:0.1 .
FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 OMM_WEIGHTS_DIR=/weights HF_HOME=/weights/hf
WORKDIR /app
RUN pip install --no-cache-dir "voxtell>=0.1.0" "huggingface_hub>=0.25" "SimpleITK>=2.3" "pillow>=9.4" "scipy>=1.10"
COPY src/open_med_mcp/zoo/_sdk/omm_job.py /app/omm_job.py
COPY src/open_med_mcp/zoo/voxtell/run.py /app/run.py
ENTRYPOINT ["python", "/app/run.py"]
CMD ["--job", "/job"]
