FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-devel

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN apt-get update --fix-missing
RUN apt-get install -y vim
RUN apt-get install -y libtiff-dev
RUN apt-get install -y libgl1-mesa-glx
RUN apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev
RUN apt-get install -y tmux vim htop wget python-setuptools curl git
RUN apt-get install -y unzip

WORKDIR /home

COPY . .

RUN pip install --upgrade pip
RUN pip install -r /home/requirements.txt
