# CUDA 11.8 + cuDNN 포함된 기본 이미지 사용
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# 시스템 필수 패키지 설치
RUN apt-get update && apt-get install -y \
    curl git nano sudo \
    libgl1-mesa-glx libglib2.0-0 libglib2.0-dev \
    libsm6 libxrender1 libxext6 ffmpeg unzip wget \
    && rm -rf /var/lib/apt/lists/*

# Miniconda 설치
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

RUN curl -sLo ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh && \
    bash ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh && \
    conda clean -afy && \
    pip install --upgrade pip && \
    conda init bash

# bash를 기본 쉘로 설정
SHELL ["/bin/bash", "-c"]

# 작업 디렉토리 생성
WORKDIR /workspace

# 기본 패키지 설치 (원한다면 여기서 더 추가 가능!)
RUN pip install opencv-python-headless matplotlib tqdm seaborn

# requirements.yaml은 마운트 영향 안 받는 /opt에 복사
COPY requirements.txt /opt/requirements.txt
RUN ln -s /opt/requirements.txt /workspace/requirements.txt


# Entrypoint에서 symlink 만들고 bash 실행
CMD ["/bin/bash", "-c", "ln -sf /opt/requirements.txt /workspace/requirements.txt && cd /workspace && exec bash"]

