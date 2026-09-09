# Alternative to render.yaml's native Python runtime, if you'd rather deploy
# via a container (Render supports both — set the service's Runtime to
# "Docker" instead of using render.yaml's `runtime: python`).
FROM python:3.12-slim

WORKDIR /app

# System deps for building any C-extension wheels (numpy/scikit-learn) that
# don't have a prebuilt wheel for this exact base image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Uvicorn binds to $PORT (Render sets this automatically for web services;
# defaults to 8000 locally — see scripts/run_forward_test.py).
EXPOSE 8000

CMD ["python", "scripts/run_forward_test.py"]
