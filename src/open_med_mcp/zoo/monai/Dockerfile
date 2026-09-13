# open-med-mcp "monai" adapter (CUDA 12.4 runtime). Bundles download on first use into /weights.
#   docker build -f src/open_med_mcp/zoo/monai/Dockerfile -t ghcr.io/d0ng231/open-med-mcp-monai:0.1 .
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 OMM_WEIGHTS_DIR=/weights
WORKDIR /app
RUN pip install --no-cache-dir "monai[nibabel,ignite]>=1.4" "SimpleITK>=2.3" "pillow>=9.4" "scipy>=1.10"
COPY src/open_med_mcp/zoo/_sdk/omm_job.py /app/omm_job.py
COPY src/open_med_mcp/zoo/monai/run.py /app/run.py
ENTRYPOINT ["python", "/app/run.py"]
CMD ["--job", "/job"]
