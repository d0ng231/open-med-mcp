# open-med-mcp "totalsegmentator" adapter (CUDA 12.4 runtime). Weights download on first use into
# /weights/totalsegmentator (mount $OMM_HOME/weights at /weights; needs internet on first run).
#   docker build -f src/open_med_mcp/zoo/totalsegmentator/Dockerfile -t ghcr.io/d0ng231/open-med-mcp-totalsegmentator:0.1 .
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 OMM_WEIGHTS_DIR=/weights TOTALSEG_HOME_DIR=/weights/totalsegmentator
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsm6 libxext6 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN pip install --no-cache-dir "TotalSegmentator>=2.4" "SimpleITK>=2.3" "pillow>=9.4"
COPY src/open_med_mcp/zoo/_sdk/omm_job.py /app/omm_job.py
COPY src/open_med_mcp/zoo/totalsegmentator/run.py /app/run.py
ENTRYPOINT ["python", "/app/run.py"]
CMD ["--job", "/job"]
