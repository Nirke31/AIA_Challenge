####################################
# EXAMPLE IMAGE FOR PYTHON SUBMISSIONS
####################################
FROM ubuntu:22.04

# Use this one for submissions that require GPU processing
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y python3 python3-pip && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# ADITIONAL PYTHON DEPENDENCIES (if you have them)
COPY requirements.txt ./
RUN pip install -r requirements.txt

WORKDIR /app

# COPY WHATEVER OTHER SCRIPTS YOU MAY NEED
COPY trained_model/ /trained_model
COPY src/ ./src/
 
 # SPECIFY THE ENTRYPOINT SCRIPT
CMD python src/submission.py

# docker build -t miserymodel .
# docker run --rm --gpus all --name miserymodel -v /home/backwelle/splid-devkit/dataset:/dataset -v/home/backwelle/splid-devkit/submission:/submission miserymodel
# evalai push miserymodel:latest --phase mit-competition-2164