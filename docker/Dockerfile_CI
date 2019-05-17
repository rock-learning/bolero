FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -qq && apt-get install -y \
  cmake \
  wget \
  sudo \
  git \
  unzip \
  python \
  python-dev \
  python-pip \
  python-yaml \
  python-matplotlib \
  python-tk \
  libboost-all-dev \
  libeigen3-dev \
  libqt4-dev \
  libtinyxml-dev \
  pkg-config \
  libyaml-cpp-dev \
  libblas-dev \
  liblapack-dev \
  gfortran \
  cython \
  python-nose \
  python-scipy \
  python-sklearn \
  libjsoncpp-dev
RUN pip install gym scikit-optimize
