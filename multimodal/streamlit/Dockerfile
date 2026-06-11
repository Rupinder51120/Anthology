FROM python:3.11-slim

WORKDIR /app

COPY multimodal/streamlit/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app

CMD ["streamlit", "run", "multimodal/streamlit/app.py", "--server.port=8502", "--server.address=0.0.0.0"]
