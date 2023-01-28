FROM ubuntu:20.04
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt install -y python3-pip libgl1-mesa-glx libsm6 libxext6 libxrender-dev libglib2.0-0
RUN pip install cython numpy

WORKDIR is-multiple-object-tracking

COPY src src
COPY etc/conf etc/conf
COPY tools tools
COPY setup.py .
COPY requirements.txt .


# RUN pip install -r requirements.txt
RUN pip install .