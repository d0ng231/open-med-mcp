FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 OMM_WEIGHTS_DIR=/weights
RUN pip install --no-cache-dir "numpy>=1.24" "scipy>=1.10" "SimpleITK>=2.3" "pillow>=9.4"
WORKDIR /app
COPY src/open_med_mcp/zoo/_sdk/omm_job.py /app/omm_job.py
COPY src/open_med_mcp/zoo/_template/run.py /app/run.py
ENTRYPOINT ["python", "/app/run.py"]
CMD ["--job", "/job"]
