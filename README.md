# Anthology

AI Research Intelligence System for exploring and querying scientific literature.

## Clone the Repository

```bash
git clone https://github.com/Rupinder51120/Anthology.git
cd Anthology
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file and configure the required API keys and database settings.

```env
DATABASE_URL=
GROQ_API_KEY=
COHERE_API_KEY=
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
```

## Run with Docker

```bash
docker compose up -d
```

## Run Locally

```bash
alembic upgrade head

uvicorn api.main:app --reload
```

API Documentation:

```text
http://localhost:8000/docs
```

## License

MIT
