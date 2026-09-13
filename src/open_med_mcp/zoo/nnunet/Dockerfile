# open-med-mcp "nnunet" adapter (CUDA 12.4 runtime). Models live under /weights/nnunet/results.
#   docker build -f src/open_med_mcp/zoo/nnunet/Dockerfile -t ghcr.io/d0ng231/open-med-mcp-nnunet:0.1 .
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 OMM_WEIGHTS_DIR=/weights nnUNet_raw=/weights/nnunet/raw nnUNet_preprocessed=/weights/nnunet/preprocessed nnUNet_results=/weights/nnunet/results
WORKDIR /app
RUN pip install --no-cache-dir "nnunetv2>=2.5" "SimpleITK>=2.3" "pillow>=9.4"
COPY src/open_med_mcp/zoo/_sdk/omm_job.py /app/omm_job.py
COPY src/open_med_mcp/zoo/nnunet/run.py /app/run.py
ENTRYPOINT ["python", "/app/run.py"]
CMD ["--job", "/job"]
