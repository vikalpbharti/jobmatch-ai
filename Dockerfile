# Container for Hugging Face Spaces (or any Docker host). Serves the web UI and API on port 7860.
FROM python:3.12-slim

RUN useradd -m -u 1000 user
WORKDIR /app

# CPU-only PyTorch keeps the image small; then the app's own dependencies.
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .
RUN chown user:user /app
USER user
ENV HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1

# Download the jobs + model and build the index at image build time, so the app starts fast.
RUN python scripts/build_index.py --limit 5000

EXPOSE 7860
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
