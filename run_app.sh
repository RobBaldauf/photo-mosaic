#!/bin/bash
export TF_CPP_MIN_LOG_LEVEL=3 # Increase tf log level to hide warnings
uvicorn photo_mosaic.app:app --host 0.0.0.0 --port 8111 --workers 1