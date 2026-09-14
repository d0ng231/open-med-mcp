# open-med-mcp "vlm" adapter: Hugging Face image-text-to-text models (MedGemma, Qwen2.5-VL ...), CUDA 12.4
#   docker build -f src/open_med_mcp/zoo/vlm/Dockerfile -t ghcr.io/d0ng231/open-med-mcp-vlm:0.1 .
# Models download on first use into /weights/vlm/hf; pass HF_TOKEN for gated models.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 OMM_WEIGHTS_DIR=/weights HF_HOME=/weights/vlm/hf
WORKDIR /app
RUN pip install --no-cache-dir "transformers>=4.50" "accelerate>=1.0" "huggingface_hub>=0.25" "SimpleITK>=2.3" "pillow>=9.4" "scipy>=1.10"
COPY src/open_med_mcp/zoo/_sdk/omm_job.py /app/omm_job.py
COPY src/open_med_mcp/zoo/vlm/run.py /app/run.py
ENTRYPOINT ["python", "/app/run.py"]
CMD ["--job", "/job"]
