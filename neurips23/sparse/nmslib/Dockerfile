FROM neurips23

RUN apt update
RUN apt-get install -y python-setuptools python-pip
RUN pip3 install nmslib

# RUN python3 -c 'import sparse_hnswlib'
WORKDIR /home/app