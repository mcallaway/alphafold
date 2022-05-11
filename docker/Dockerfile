# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG CUDA=11.1
FROM nvidia/cuda:${CUDA}-cudnn8-runtime-ubuntu18.04
# FROM directive resets ARGS, so we specify again (the value is retained if
# previously set).
ARG CUDA

# We need to set the http_proxy environment variable *within* the
# container during the build process. This environment variable
# includes authentication credentials, which should not be saved.
# Here we use an ARG to set the ENV to be used in the target file.
ARG proxy
ENV http_proxy=$proxy

# Use bash to support string substitution.
SHELL ["/bin/bash", "-c"]

# Add the Monsanto SSL certs from 
# git@github.platforms.engineering:CloudOps/SSL-TLS-Configuration.git
# Start with monsanto_all.pem and split it with:
#   csplit -f cert- ../monsanto_all.pem '/-----BEGIN CERTIFICATE-----/' '{*}'
# Then make sure they all end with .crt (not .pem).
COPY docker/bayer/* /usr/local/share/ca-certificates/

# We need to get keys from nvidia in order to make use of their apt repo.
RUN echo -ne "Acquire::http::proxy \"$http_proxy\";\n" > /etc/apt/apt.conf.d/40proxy && \
    cat /etc/apt/apt.conf.d/40proxy | od -c

RUN mv /etc/apt/sources.list.d/cuda.list /tmp && \
    mv /etc/apt/sources.list.d/nvidia-ml.list /tmp && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y curl ca-certificates && \
    curl -o - http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub | apt-key add && \
    curl -o - http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub | apt-key add && \
    mv /tmp/cuda.list /etc/apt/sources.list.d/ && \
    mv /tmp/nvidia-ml.list /etc/apt/sources.list.d/ && \
    update-ca-certificates

RUN apt-get update && \
      DEBIAN_FRONTEND=noninteractive apt-get install -y \
      build-essential \
      cmake \
      cuda-command-line-tools-${CUDA/./-} \
      git \
      hmmer \
      kalign \
      tzdata \
      wget \
    && rm -rf /var/lib/apt/lists/*

# Compile HHsuite from source.
# Note that the target Git repo no longer has branches like v3.3.0
# So build with the master branch.
#RUN git clone --branch v3.3.0 https://github.com/soedinglab/hh-suite.git /tmp/hh-suite \
RUN git clone https://github.com/soedinglab/hh-suite /tmp/hh-suite \
    && mkdir /tmp/hh-suite/build \
    && pushd /tmp/hh-suite/build \
    && cmake -DCMAKE_INSTALL_PREFIX=/opt/hhsuite .. \
    && make -j 4 && make install \
    && ln -s /opt/hhsuite/bin/* /usr/bin \
    && popd \
    && rm -rf /tmp/hh-suite

# Install Miniconda package manager.
RUN wget -q -P /tmp \
  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash /tmp/Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda \
    && rm /tmp/Miniconda3-latest-Linux-x86_64.sh

# Install conda packages.
# Add mamba, a much faster dependency solver.
ENV PATH="/opt/conda/bin:$PATH"
RUN conda update -qy conda \
    && conda install -c conda-forge mamba \
    && mamba install -y -c conda-forge \
      openmm=7.5.1 \
      cudatoolkit==${CUDA_VERSION} \
      pdbfixer \
      pip \
      python=3.7

COPY . /app/alphafold
RUN wget -q -P /app/alphafold/alphafold/common/ \
  https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt

# Install pip packages.
RUN pip3 install --upgrade pip \
    && pip3 install -r /app/alphafold/requirements.txt \
    && pip3 install --upgrade jax==0.2.14 jaxlib==0.1.69+cuda${CUDA/./} -f \
      https://storage.googleapis.com/jax-releases/jax_releases.html

# Apply OpenMM patch.
WORKDIR /opt/conda/lib/python3.7/site-packages
RUN patch -p0 < /app/alphafold/docker/openmm.patch

# Add SETUID bit to the ldconfig binary so that non-root users can run it.
RUN chmod u+s /sbin/ldconfig.real

# We need to run `ldconfig` first to ensure GPUs are visible, due to some quirk
# with Debian. See https://github.com/NVIDIA/nvidia-docker/issues/1399 for
# details.
# ENTRYPOINT does not support easily running multiple commands, so instead we
# write a shell script to wrap them up.
WORKDIR /app/alphafold
RUN echo $'#!/bin/bash\n\
ldconfig\n\
python /app/alphafold/run_alphafold.py "$@"' > /app/run_alphafold.sh \
  && chmod +x /app/run_alphafold.sh
ENTRYPOINT ["/app/run_alphafold.sh"]
