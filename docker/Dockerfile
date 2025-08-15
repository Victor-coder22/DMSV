#syntax=docker/dockerfile:1
#FROM docker.io/tensorflow/tensorflow:2.11.1-gpu
#FROM python:3.10-bookworm
FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ENV TZ=Europe/Berlin
#dont run as root (end upa and cont create cashfile error)
RUN groupadd -g 1000 appuser && useradd --system -u 1000 -g appuser -m appuser
# install cuda toolkit
RUN apt update && apt install -y cuda-toolkit-11-2 cudnn
#install python
RUN apt update && apt install -y python3 python3-doc python3-pip 
#install python dependencies
RUN pip install --upgrade pip
# install the tensorflow (requires special numpy version) with cuda
RUN pip install tensorflow[and-cuda]==2.11.1 numpy==1.24.2
#RUN pip install tensorflow[and-cuda]
# install projet depend packages
RUN pip install notebook
RUN pip install scikit-learn
RUN pip install natsort
RUN pip install pillow
RUN pip install matplotlib 
RUN pip install papermill
RUN pip install pycocotools
RUN pip install focal-loss
RUN pip install opencv-python-headless

# install xelatex for matplotlib pgf backend
#RUN apt update && apt install -y texlive-xetex

USER appuser
WORKDIR /tf

#jupyther notebook
CMD ["bash", "-c", "source /etc/bash.bashrc && jupyter notebook --notebook-dir=/tf --ip 0.0.0.0 --no-browser"]

