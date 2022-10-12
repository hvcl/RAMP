FROM tensorflow/tensorflow:2.6.0rc1-gpu

RUN apt-key del 7fa2af80 && apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/3bf863cc.pub
RUN apt-get update
RUN apt-get install -y vim
RUN apt-get install -y libtiff-dev
RUN apt-get install -y libgl1-mesa-glx
RUN apt-get install -y tmux vim htop wget python-setuptools curl git sudo

WORKDIR /home

COPY . .

RUN pip install --upgrade pip
RUN pip install -r /home/requirements.txt


