#!/bin/bash

export PYTHONPATH=`pwd`

CMAKE_FLAGS="-DBUILD_TESTING=off"

if [ "x$1" == "xtest" ] ; then
    CMAKE_FLAGS="${CMAKE_FLAGS} -DBUILD_COVERAGE=ON"
fi

cd $(dirname ${BASH_SOURCE[0]})

set -e

# build zookeeper
(cd third_party/zookeeper; ant compile_jute)

rm -fr build && mkdir build
(cd build && cmake ${CMAKE_FLAGS} .. && make -j32)

rm -f tools/remote_console
ln -s $(pwd)/build/euler/tools/remote_console/remote_console tools/

python tools/pip/setup.py install

if [ "x$1" == "xtest" ] ; then
   echo "Generating test data ----"
   python euler/tools/generate_euler_data.py tools/test_data/graph.json /tmp/euler 2 tools/test_data/meta

  cd build && make test
fi

