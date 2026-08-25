FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV TOKENIZERS_PARALLELISM=false

RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# rapidocr (a transitive docling dependency) pulls in opencv-python, which
# needs libGL. opencv-python-headless is the same OpenCV build without the
# GUI/video-I/O bindings docling/rapidocr never call — force-reinstalling it
# here (rather than editing requirements.txt) deterministically wins over
# whatever pip's resolver installs for the unpinned transitive opencv-python.
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt \
    && pip install --no-cache-dir --no-deps --force-reinstall opencv-python-headless==4.11.0.86

COPY . .

RUN chmod +x start.sh

EXPOSE 8000

CMD ["./start.sh"]
