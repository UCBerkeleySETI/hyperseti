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
RUN cd test && python3 download_test_data.py && cd ..
RUN cd test && python3 -m pytest . && cd ..

RUN find test -name "*.h5" -type f -delete
RUN find test -name "*.log" -type f -delete
RUN find test -name "*.dat" -type f -delete
RUN find test -name "*.fil" -type f -delete
RUN find test -name "*.png" -type f -delete
RUN find . -path '*/__pycache__*' -delete

WORKDIR /home
