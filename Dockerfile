FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml requirements.txt ./
COPY src ./src
COPY schemas ./src/fade_content_agent/schemas
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir .

ENV PORT=8080
EXPOSE 8080
CMD ["sh", "-c", "uvicorn fade_content_agent.api:app --host 0.0.0.0 --port ${PORT}"]
