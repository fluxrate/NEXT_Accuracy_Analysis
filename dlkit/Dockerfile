FROM nvidia/cuda:cudnn
RUN apt-get update && apt-get install -y curl python
RUN curl https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py && python /tmp/get-pip.py && rm /tmp/get-pip.py

RUN curl https://root.cern.ch/download/root_v6.08.06.Linux-ubuntu16-x86_64-gcc5.4.tar.gz -o /opt/root.tar.gz && tar -xzvf /opt/root.tar.gz -C /opt

COPY ./ /opt/DLKit/

RUN /bin/bash -c "source /opt/root/bin/thisroot.sh && pip install --no-cache-dir -r /opt/DLKit/requirements.txt"
