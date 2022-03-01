FROM ubuntu:20.04
LABEL maintaner="Lucas Nunes<lucas.nunes@uni-bonn.de>"

RUN apt-get update && apt-get install --no-install-recommends -y \
    ca-certificates \
    zsh \
    wget \
    libusb-1.0-0-dev \
    libgl-dev \
    python3-dev \
    python3-pip \
    build-essential \
    libopenblas-dev \
    nvidia-cuda-toolkit \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install \
    matplotlib==3.3.4 \
    numpy==1.20.1 \
    open3d==0.12.0 \
    plyfile==0.7.3 \
    pytorch-lightning==1.1.8 \
    tensorboard==2.4.1 \
    tensorboard-plugin-wit==1.8.0 \
    torch==1.7.1 \
    torchvision==0.8.2 \
    tqdm==4.56.2 \
    h5py==3.6.0 \ 
    ninja==1.10.2.3 \
    hdbscan

#RUN pip3 install -U MinkowskiEngine==0.5.4 --install-option="--blas=openblas" -v --no-deps

RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.1/zsh-in-docker.sh)" -- -t gentoo
CMD ["zsh"]

