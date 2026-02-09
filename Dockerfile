FROM python:3.12-slim

WORKDIR /app

COPY repository_after/ ./repository_after/
COPY tests/ ./tests/

ENV PYTHONPATH=/app/repository_after:$PYTHONPATH

CMD ["python", "tests/test_tensor.py"]
