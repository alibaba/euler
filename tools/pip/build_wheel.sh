#!/usr/bin/env bash
# Copyright 2018 Alibaba Group Holding Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

set -e

TF_VERSION="1.12.0"

POSITIONAL=()
while [[ $# -gt 0 ]]
do
  key="$1"
  case $key in
    --tf_version)
    TF_VERSION="$2"
    shift
    shift
    ;;
    *)
    POSITIONAL+=("$1")
    shift
    ;;
  esac
done
set -- "${POSITIONAL[@]}"

curl http://mirrors.aliyun.com/repo/Centos-7.repo -o /etc/yum.repos.d/CentOS-Base.repo
curl http://mirrors.aliyun.com/repo/epel-7.repo -o /etc/yum.repos.d/epel.repo
echo "[global]
index-url = https://mirrors.aliyun.com/pypi/simple/" > /etc/pip.conf

yum install -y autoconf ant cmake3 gcc-c++ golang java-1.8.0-openjdk-headless perf python2-pip make

pip install tensorflow=="$TF_VERSION"

curl -O https://mirrors.aliyun.com/apache/hadoop/common/hadoop-2.9.2/hadoop-2.9.2.tar.gz
tar xf hadoop-2.9.2.tar.gz -C /usr/local
export LIBRARY_PATH="/usr/local/hadoop-2.9.2/lib/native:$LIBRARY_PATH"
export LD_LIBRARY_PATH="/usr/local/hadoop-2.9.2/lib/native:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/usr/lib/jvm/jre/lib/amd64/server:$LD_LIBRARY_PATH"

SCRIPT_DIR=$(dirname "$0")
PROJECT_DIR="$SCRIPT_DIR/../.."
mkdir -p /build
cd /build
cmake3 "$PROJECT_DIR"
make -j $(expr $(nproc) \* 2)
cd "$PROJECT_DIR"
python tools/pip/setup.py bdist_wheel --python-tag cp27 -p manylinux1_x86_64 --tf_version "$TF_VERSION"
