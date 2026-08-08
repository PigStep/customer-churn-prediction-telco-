FROM python:3.14-slim
WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update && \ 
        apt-get install -y --no-install-recommends libgomp1
        
COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-install-project

# Skip dockerignore files
COPY . . 

EXPOSE 8501

CMD ["uv", "run" ,"streamlit", "run", "streamlit_app.py"]