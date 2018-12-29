FROM ubuntu:18.04

COPY tools/docker/sources.list /etc/apt
RUN apt-get update && apt-get install -y --no-install-recommends \
    ant \
    autoconf \
    build-essential \
    cmake \
    default-jre \
    golang-go \
    python \
    python-pip \
    python-setuptools \
    wget \
    && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME /usr/lib/jvm/java-11-openjdk-amd64
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:$JAVA_HOME/lib/server

RUN wget https://mirrors.aliyun.com/apache/hadoop/common/hadoop-2.9.2/hadoop-2.9.2.tar.gz && \
    tar xf hadoop-2.9.2.tar.gz -C /usr/local && \
    rm -rf hadoop-2.9.2.tar.gz
ENV HADOOP_HOME /usr/local/hadoop-2.9.2
ENV LIBRARY_PATH $LIBRARY_PATH:$HADOOP_HOME/lib/native
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:$HADOOP_HOME/lib/native
ENV PATH $PATH:$HADOOP_HOME/bin
RUN sh -c 'echo export CLASSPATH=$CLASSPATH:$(hadoop classpath --glob) >> /etc/bash.bashrc'

RUN mkdir -p /root/.pip
COPY tools/docker/pip.conf /root/.pip
RUN pip --no-cache-dir install tensorflow

COPY . /tmp/Euler
RUN cd /tmp/Euler/third_party/zookeeper && \
    ((cd zookeeper-client/zookeeper-client-c; \
      [ -e generated/zookeeper.jute.h ] && [ -e generated/zookeeper.jute.c ]) || \
     ant compile_jute) && \
    cd /tmp/Euler && \
    mkdir -p /tmp/build && cd /tmp/build && \
    cmake /tmp/Euler && \
    make -j $(expr $(nproc) \* 2) && \
    cd /tmp/Euler && \
    python tools/pip/setup.py install && \
    rm -rf /tmp/Euler /tmp/build
