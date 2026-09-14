# open-med-mcp "radiomics" adapter: pyradiomics 3.1 (needs Python 3.9 + NumPy 1.x), CPU only, ~350 MB
#   docker build -f src/open_med_mcp/zoo/radiomics/Dockerfile -t ghcr.io/d0ng231/open-med-mcp-radiomics:0.1 .
FROM python:3.9-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN pip install --no-cache-dir "numpy<2" "pyradiomics==3.1.0" "SimpleITK>=2.3" "pillow>=9.4" "scipy<1.14"
WORKDIR /app
COPY src/open_med_mcp/zoo/_sdk/omm_job.py /app/omm_job.py
COPY src/open_med_mcp/zoo/radiomics/run.py /app/run.py
ENTRYPOINT ["python", "/app/run.py"]
CMD ["--job", "/job"]
