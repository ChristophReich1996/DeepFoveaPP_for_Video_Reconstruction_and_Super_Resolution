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
# Intall required python packeges
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
# Set working directory
WORKDIR /workspace/repositories
# Set python path
ENV PYTHONPATH "${PYTHONPATH}:./"

