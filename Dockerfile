FROM cupy/cupy:v9.0.0

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update

COPY . /turboseti
WORKDIR /turboseti

# Install system requirements.
RUN cat requirements_sys.txt | xargs -n 1 apt install --no-install-recommends -y

# Install Python requirements.
RUN python3 -m pip install -U pip
RUN python3 -m pip install $(grep -ivE "cupy-cuda110|cupy" requirements.txt)
RUN python3 -m pip install -r requirements_test.txt
RUN python3 setup.py install

WORKDIR /home