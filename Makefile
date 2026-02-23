.PHONY: run build docker-run test

run:
    uvicorn app.main:app --reload

build:
    docker build -t wifi-signal-api .

docker-run:
    docker run -p 8000:8000 wifi-signal-api

test:
    pytest
