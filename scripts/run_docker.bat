#!/bin/bash

echo building docker images and starting docker containers...
docker compose up --build -d
daphne -b 0.0.0.0 -p 8000 image_labeling_backend.asgi:application