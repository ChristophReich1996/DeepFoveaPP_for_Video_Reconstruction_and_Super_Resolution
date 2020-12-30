# Get pytorch docker
FROM nvcr.io/nvidia/pytorch:20.12-py3
# Perform update
RUN ["apt-get", "update"]
# Install zsh
RUN ["apt-get", "install", "-y", "zsh"]
# Install wget
RUN ["apt-get", "install", "-y", "wget"]
# Install curl
RUN ["apt-get", "install", "-y", "curl"]
# Install git
RUN ["apt-get", "install", "-y", "git"]
# Intall tmux
RUN ["apt-get", "install", "-y", "tmux"]
# Install older GCC compiler
RUN ["apt-get", "install", "-y", "gcc-6.3.0"]
RUN ["apt-get", "install", "-y", "g++-6.3.0"]
# Intall required python packeges
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
# Install pade activation units
WORKDIR /workspace/repositories/DeepFoveaPP_for_Video_Reconstruction_and_Super_Resolution/pade_activation_unit/cuda
RUN python setup.py install
# Set working directory
WORKDIR /workspace/repositories
# Set python path
ENV PYTHONPATH "${PYTHONPATH}:./"

