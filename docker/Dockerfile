ARG PYTORCH="1.6.0"
ARG CUDA="10.1"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 openssh-server \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/* \
 && echo "Port 22                     #开启22端口" >> /etc/ssh/sshd_config \
 && echo "PermitRootLogin yes         #允许root用户使用ssh登录" >> /etc/ssh/sshd_config \
 && echo "PubkeyAuthentication yes    #启用公钥私钥配对认证方式" >> /etc/ssh/sshd_config \
 && echo "AuthorizedKeysFile          .ssh/authorized_keys .ssh/authorized_keys2 " >> /etc/ssh/sshd_config


# Install mmsegmentation
RUN conda clean --all

RUN pip install mmcv-full==latest+torch1.6.0+cu101 -f https://download.openmmlab.com/mmcv/dist/index.html
RUN git clone https://github.com/CarnoZhao/mmsegmentation.git /mmsegmentation
WORKDIR /mmsegmentation
RUN git checkout v0.17.0.base
RUN pip install -r requirements.txt
RUN pip install --no-cache-dir -e .
