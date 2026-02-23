# Wireless Signal Quality API

A Dockerized FastAPI microservice that classifies WiFi signal quality (Excellent/Good/Fair/Poor) from RF metrics using a trained ML model.

## Problem

In wireless engineering, we constantly reason about signal quality from metrics like RSSI, SNR, channel width, and band. This project turns that reasoning into a small, reproducible ML-backed API.

## Features

- FastAPI service with `/health` and `/predict` endpoints
- ML model (RandomForest) trained on synthetic WiFi-like data
- Model serialized with `joblib` and loaded at startup
- Dockerized for reproducible deployment
- Basic tests with `pytest`

## Tech Stack

- Python, FastAPI, Pydantic
- scikit-learn, joblib
- Docker
- pytest

## Project Structure

```text
app/        # API, schemas, model loader
models/     # Serialized ML model
data/       # Synthetic training data
tests/      # API tests

